"""The public surface other projects depend on. Breaking these breaks them."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import graph_looper
from graph_looper import (
    Graph,
    GraphError,
    MemoryStateStore,
    MockProvider,
    RunError,
    arun,
    available,
    lint,
    load,
    run,
    to_mermaid,
    validate,
)

DOCUMENT = """
name: inline
nodes:
  - id: start
    type: input
  - id: work
    type: agent
    prompt: "handle {{ task }}"
  - id: done
    type: output
    prompt: "{{ inputs.work }}"
edges:
  - from: start
    to: work
  - from: work
    to: done
"""


# -- load() takes whatever the caller already has --------------------------


def test_load_accepts_a_bundled_name():
    assert load("reviewer-loop").name == "reviewer-loop"


def test_load_accepts_a_path_string(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text(DOCUMENT)
    assert load(str(path)).name == "inline"


def test_load_accepts_a_path_object(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text(DOCUMENT)
    assert load(path).name == "inline"


def test_load_accepts_yaml_text():
    assert load(DOCUMENT).name == "inline"


def test_load_accepts_a_dict():
    assert load(load(DOCUMENT).to_dict()).name == "inline"


def test_load_passes_a_graph_through_unchanged():
    graph = load(DOCUMENT)
    assert load(graph) is graph


def test_load_rejects_an_unsupported_type():
    with pytest.raises(GraphError) as excinfo:
        load(42)
    assert "takes a Graph" in str(excinfo.value)


def test_load_names_what_is_available_when_it_cannot_find_a_graph():
    with pytest.raises(GraphError) as excinfo:
        load("no-such-graph")
    assert "reviewer-loop" in str(excinfo.value)


def test_load_searches_extra_directories(tmp_path):
    (tmp_path / "mine.yaml").write_text(DOCUMENT)
    assert load("mine", search_paths=[tmp_path]).name == "inline"


# -- run() -----------------------------------------------------------------


def test_the_one_liner():
    result = run(DOCUMENT, "the task", provider=MockProvider({"work": ["done it"]}))
    assert result.ok
    assert result.output == "done it"


def test_run_reports_failure_without_raising():
    """A failed run comes back as a result with `ok=False`, not an exception."""
    stalls = DOCUMENT.replace(
        '    prompt: "handle {{ task }}"',
        '    prompt: x\n    label_from: data.kind',
    ).replace("  - from: work\n    to: done", "  - from: work\n    to: done\n    when: never")
    result = run(stalls, "t", provider=MockProvider({"work": [{"kind": "other"}]}))
    assert not result.ok
    assert "stalled" in result.error


def test_raise_for_status_turns_a_failure_into_an_exception():
    graph = DOCUMENT.replace(
        "  - id: done\n    type: output",
        "  - id: g\n    type: gate\n    choices: [a, b]\n    on_exhausted: a\n"
        "    prompt: p\n  - id: done\n    type: output",
    ).replace(
        "  - from: work\n    to: done",
        "  - from: work\n    to: g\n  - from: g\n    to: done\n    when: a\n"
        "  - from: g\n    to: done\n    when: b",
    )
    result = run(graph, "t", provider=MockProvider({"g": [{"choice": "bad"}]}))
    assert not result.ok
    with pytest.raises(RunError):
        result.raise_for_status()


def test_variables_reach_prompts():
    document = DOCUMENT.replace("handle {{ task }}", "handle {{ vars.who }}")
    provider = MockProvider()
    run(document, "t", provider=provider, variables={"who": "Cal"})
    assert provider.calls_for("work")[0].prompt == "handle Cal"


def test_on_event_streams_progress():
    seen: list[str] = []
    run(DOCUMENT, "t", provider=MockProvider(), on_event=lambda e: seen.append(e.kind))
    assert seen[0] == "run_start"
    assert "node_end" in seen


def test_arun_works_inside_an_existing_event_loop():
    async def main():
        return await arun(DOCUMENT, "t", provider=MockProvider({"work": ["async"]}))

    assert asyncio.run(main()).output == "async"


def test_state_flows_through_the_convenience_function():
    store = MemoryStateStore()
    run(DOCUMENT, "t", provider=MockProvider(), state=store, namespace="ns")
    assert store.load("ns")["_runs"] == 1


# -- result shape ----------------------------------------------------------


def test_result_accessors():
    result = run(DOCUMENT, "t", provider=MockProvider({"work": ["text here"]}))
    assert result.text_of("work") == "text here"
    assert result.text_of("absent", "fallback") == "fallback"
    assert result.data_of("absent") is None
    assert result.total_tokens == result.input_tokens + result.output_tokens


def test_usage_is_counted_across_every_visit_not_just_the_last():
    """A node in a loop bills more than once; the totals must say so."""
    from graph_looper import load_graph_str
    from tests.conftest import LOOP

    graph = load_graph_str(LOOP)
    result = run(
        graph,
        "t",
        provider=MockProvider(
            {
                "worker": ["one", "two"],
                "check": [{"choice": "fail", "reason": "no"}, {"choice": "pass", "reason": "ok"}],
            }
        ),
    )
    assert result.usage_by_node["worker"]["calls"] == 2
    assert result.usage_by_node["worker"]["model_calls"] == 2


def test_costliest_ranks_the_spenders():
    result = run(DOCUMENT, "t", provider=MockProvider({"work": ["a" * 400]}))
    top = result.costliest(1)
    assert top[0][0] == "work"


# -- the rest of the surface ----------------------------------------------


def test_validate_returns_the_graph_or_raises():
    assert isinstance(validate(DOCUMENT), Graph)
    with pytest.raises(GraphError):
        validate("name: x\nnodes: []\n")


def test_lint_takes_the_same_sources_as_load():
    assert lint("reviewer-loop") == []


def test_to_mermaid_renders_from_any_source():
    assert to_mermaid(DOCUMENT).startswith("flowchart LR")
    assert "flowchart TB" in to_mermaid("reviewer-loop", direction="TB")


def test_to_mermaid_can_overlay_a_run():
    result = run(DOCUMENT, "t", provider=MockProvider())
    assert "stroke:#2f8f4e" in to_mermaid(DOCUMENT, trace=result.trace)


def test_available_lists_graphs_by_name():
    names = available()
    assert "reviewer-loop" in names
    assert isinstance(names["reviewer-loop"], Path)


# -- stability -------------------------------------------------------------


def test_everything_in_all_is_importable():
    for name in graph_looper.__all__:
        assert hasattr(graph_looper, name), f"__all__ promises {name} but it is missing"


def test_the_documented_surface_is_present():
    """Downstream projects import these by name; removing one is a breaking
    change and should fail here first."""
    expected = {
        "run", "arun", "load", "validate", "lint", "to_mermaid", "available",
        "Graph", "Node", "Edge", "Runner", "RunResult", "NodeResult",
        "Provider", "AnthropicProvider", "MockProvider",
        "StateStore", "FileStateStore", "MemoryStateStore", "NullStateStore",
        "GraphError", "RunError", "ProviderError", "StateError",
    }
    assert expected <= set(graph_looper.__all__)


def test_version_is_exposed():
    assert graph_looper.__version__.count(".") == 2
