from __future__ import annotations

import json

import pytest

from graph_looper import MockProvider, run
from graph_looper.state import (
    FileStateStore,
    MemoryStateStore,
    NullStateStore,
    StateError,
    record,
    resolve_store,
)

SAVES = """
name: saver
nodes:
  - id: start
    type: input
  - id: note
    type: agent
    writes_state: lessons
    state_append: true
    state_limit: 3
    prompt: "prior: {{ state.lessons }}"
  - id: done
    type: output
    prompt: "{{ inputs.note }}"
edges:
  - from: start
    to: note
  - from: note
    to: done
"""


def test_null_store_remembers_nothing():
    store = NullStateStore()
    store.save("g", {"a": 1})
    assert store.load("g") == {}


def test_memory_store_round_trips_and_isolates_namespaces():
    store = MemoryStateStore()
    store.save("one", {"a": 1})
    store.save("two", {"b": 2})
    assert store.load("one") == {"a": 1}
    assert store.load("two") == {"b": 2}
    assert store.load("three") == {}


def test_memory_store_copies_so_callers_cannot_mutate_it():
    store = MemoryStateStore()
    store.save("g", {"a": [1]})
    loaded = store.load("g")
    loaded["a"] = "clobbered"
    assert store.load("g") == {"a": [1]}


def test_file_store_persists_across_instances(tmp_path):
    path = tmp_path / "nested" / "state.json"
    FileStateStore(path).save("g", {"a": 1})
    assert path.exists()
    assert FileStateStore(path).load("g") == {"a": 1}


def test_file_store_keeps_other_namespaces_when_saving(tmp_path):
    path = tmp_path / "state.json"
    store = FileStateStore(path)
    store.save("one", {"a": 1})
    store.save("two", {"b": 2})
    assert store.load("one") == {"a": 1}
    assert json.loads(path.read_text()).keys() == {"one", "two"}


def test_file_store_reports_corrupt_json_clearly(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    with pytest.raises(StateError) as excinfo:
        FileStateStore(path).load("g")
    assert "not valid JSON" in str(excinfo.value)


def test_file_store_rejects_a_json_list(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("[]")
    with pytest.raises(StateError):
        FileStateStore(path).load("g")


def test_missing_file_is_empty_not_an_error(tmp_path):
    assert FileStateStore(tmp_path / "absent.json").load("g") == {}


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, NullStateStore),
        (False, NullStateStore),
        (True, FileStateStore),
        ("some/path.json", FileStateStore),
    ],
)
def test_resolve_store_accepts_the_documented_shapes(value, expected):
    assert isinstance(resolve_store(value), expected)


def test_resolve_store_passes_a_store_through():
    store = MemoryStateStore()
    assert resolve_store(store) is store


def test_resolve_store_rejects_nonsense():
    with pytest.raises(StateError):
        resolve_store(42)


def test_record_replaces_by_default():
    state = {"k": "old"}
    record(state, "k", "new")
    assert state["k"] == "new"


def test_record_appends_and_trims_to_the_limit():
    state: dict = {}
    for value in ["a", "b", "c", "d"]:
        record(state, "log", value, append=True, limit=3)
    assert state["log"] == ["b", "c", "d"]


def test_record_promotes_an_existing_scalar_to_a_list():
    state = {"log": "first"}
    record(state, "log", "second", append=True)
    assert state["log"] == ["first", "second"]


# -- integration with a run ------------------------------------------------


def test_a_node_writes_state_and_the_next_run_reads_it():
    store = MemoryStateStore()
    first = run(SAVES, "x", provider=MockProvider({"note": ["lesson one"]}), state=store)
    assert first.state["lessons"] == ["lesson one"]

    provider = MockProvider({"note": ["lesson two"]})
    second = run(SAVES, "x", provider=provider, state=store)
    assert second.state["lessons"] == ["lesson one", "lesson two"]
    # The second run's prompt was rendered with the first run's lesson in it.
    assert "lesson one" in provider.calls_for("note")[0].prompt


def test_append_limit_bounds_what_accumulates():
    store = MemoryStateStore()
    for index in range(5):
        run(SAVES, "x", provider=MockProvider({"note": [f"lesson {index}"]}), state=store)
    assert store.load("saver")["lessons"] == ["lesson 2", "lesson 3", "lesson 4"]


def test_engine_records_run_metadata():
    store = MemoryStateStore()
    run(SAVES, "x", provider=MockProvider(), state=store)
    result = run(SAVES, "x", provider=MockProvider(), state=store)
    assert result.state["_runs"] == 2
    assert result.state["_last_ok"] is True


def test_state_is_saved_even_when_the_run_fails():
    """A run that dies still learned whatever its finished nodes wrote."""
    graph = SAVES.replace(
        "  - id: done\n    type: output\n    prompt: \"{{ inputs.note }}\"",
        "  - id: gate\n    type: gate\n    choices: [a, b]\n    on_exhausted: a\n"
        "    prompt: pick\n  - id: done\n    type: output",
    ).replace(
        "  - from: note\n    to: done",
        "  - from: note\n    to: gate\n  - from: gate\n    to: done\n    when: a\n"
        "  - from: gate\n    to: done\n    when: b",
    )
    store = MemoryStateStore()
    result = run(
        graph,
        "x",
        provider=MockProvider({"note": ["saved anyway"], "gate": [{"choice": "nope"}]}),
        state=store,
    )
    assert not result.ok
    assert store.load("saver")["lessons"] == ["saved anyway"]
    assert store.load("saver")["_last_ok"] is False


def test_no_state_by_default_so_a_library_caller_gets_no_surprise_writes(tmp_path, monkeypatch):
    """Without a store nothing is written anywhere — but the result still shows
    what the run *would* have recorded, so callers can inspect it."""
    monkeypatch.chdir(tmp_path)
    result = run(SAVES, "x", provider=MockProvider())
    assert not list(tmp_path.iterdir())
    assert result.state["lessons"] == ["[mock:note]"]


def test_namespace_lets_one_store_back_several_callers():
    store = MemoryStateStore()
    run(SAVES, "x", provider=MockProvider({"note": ["a"]}), state=store, namespace="alice")
    run(SAVES, "x", provider=MockProvider({"note": ["b"]}), state=store, namespace="bob")
    assert store.load("alice")["lessons"] == ["a"]
    assert store.load("bob")["lessons"] == ["b"]


def test_reserved_keys_are_rejected_in_a_graph():
    from graph_looper.spec import GraphError, load_graph_str

    with pytest.raises(GraphError) as excinfo:
        load_graph_str(SAVES.replace("writes_state: lessons", "writes_state: _runs"))
    assert "reserved" in str(excinfo.value)
