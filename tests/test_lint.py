from __future__ import annotations

from graph_looper import lint
from graph_looper.spec import load_graph_str

BASE = """
name: t
nodes:
  - id: start
    type: input
  - id: a
    type: agent
    prompt: "do {{ task }}"
  - id: b
    type: agent
    prompt: "{{ REF }}"
  - id: done
    type: output
    prompt: "{{ results.b }}"
edges:
  - from: start
    to: a
  - from: a
    to: b
  - from: b
    to: done
"""


def codes(document: str) -> list[str]:
    return [w.code for w in lint(load_graph_str(document))]


def test_a_clean_graph_warns_about_nothing():
    assert codes(BASE.replace("{{ REF }}", "{{ inputs.a }}")) == []


def test_a_node_nobody_reads_is_flagged():
    warnings = lint(load_graph_str(BASE.replace("{{ REF }}", "no reference at all")))
    assert [w.code for w in warnings] == ["unreferenced-output"]
    assert warnings[0].node == "a"
    assert "inputs.a" in warnings[0].message


def test_the_joined_input_shortcut_counts_as_reading_every_upstream():
    """`{{ input }}` splices in all incoming messages, so `a` is read."""
    assert codes(BASE.replace("{{ REF }}", "{{ input }}")) == []


def test_reading_via_results_counts_too():
    assert codes(BASE.replace("{{ REF }}", "{{ results.a }}")) == []


def test_reading_a_structured_field_counts_too():
    assert codes(BASE.replace("{{ REF }}", "{{ data.a.verdict }}")) == []


def test_a_reference_with_no_edge_behind_it_is_flagged():
    """`{{ inputs.x }}` only fills from an incoming edge — otherwise it is
    permanently blank and the prompt silently loses a section."""
    document = BASE.replace("{{ REF }}", "{{ inputs.done }} {{ inputs.a }}")
    warnings = [w for w in lint(load_graph_str(document)) if w.code == "dangling-input"]
    assert len(warnings) == 1
    assert "'done'" in warnings[0].message
    assert "results.done" in warnings[0].message


def test_a_reference_to_a_node_that_does_not_exist_is_flagged():
    document = BASE.replace("{{ REF }}", "{{ inputs.a }} {{ inputs.ghost }}")
    warnings = [w for w in lint(load_graph_str(document)) if w.code == "unknown-node-reference"]
    assert len(warnings) == 1
    assert "ghost" in warnings[0].message


def test_gates_are_not_expected_to_be_read():
    """A gate's output is its routing decision; nothing needs to quote it."""
    document = """
    name: t
    nodes:
      - id: start
        type: input
      - id: g
        type: gate
        choices: [yes_, no_]
        on_exhausted: yes_
        prompt: "{{ task }}"
      - id: done
        type: output
    edges:
      - from: start
        to: g
      - from: g
        to: done
        when: yes_
      - from: g
        to: done
        when: no_
    """
    assert codes(document) == []


def test_writing_to_state_counts_as_a_use():
    document = BASE.replace("{{ REF }}", "unrelated").replace(
        "  - id: a\n    type: agent\n", "  - id: a\n    type: agent\n    writes_state: notes\n"
    )
    assert "unreferenced-output" not in codes(document)


def test_state_written_but_never_read_is_flagged():
    document = BASE.replace("{{ REF }}", "{{ inputs.a }}").replace(
        "  - id: a\n    type: agent\n", "  - id: a\n    type: agent\n    writes_state: notes\n"
    )
    warnings = [w for w in lint(load_graph_str(document)) if w.code == "write-only-state"]
    assert len(warnings) == 1
    assert "later runs will not benefit" in warnings[0].message


def test_state_that_is_read_back_is_fine():
    document = BASE.replace("{{ REF }}", "{{ inputs.a }} {{ state.notes }}").replace(
        "  - id: a\n    type: agent\n", "  - id: a\n    type: agent\n    writes_state: notes\n"
    )
    assert codes(document) == []


def test_an_unused_var_is_flagged():
    document = BASE.replace("{{ REF }}", "{{ inputs.a }}").replace(
        "name: t", "name: t\nvars:\n  tone: dry\n  unused: yes"
    )
    warnings = [w for w in lint(load_graph_str(document)) if w.code == "unused-var"]
    assert {w.message.split()[0] for w in warnings} == {"vars.tone", "vars.unused"}


def test_a_var_that_is_read_is_not_flagged():
    document = BASE.replace("{{ REF }}", "{{ inputs.a }} {{ vars.tone }}").replace(
        "name: t", "name: t\nvars:\n  tone: dry"
    )
    assert codes(document) == []


def test_system_prompts_count_as_references():
    document = BASE.replace('    prompt: "{{ REF }}"', '    system: "{{ inputs.a }}"\n    prompt: go')
    assert codes(document) == []


def test_warnings_render_readably():
    warning = lint(load_graph_str(BASE.replace("{{ REF }}", "nothing")))[0]
    assert str(warning).startswith("a: ")
    assert str(warning).endswith("[unreferenced-output]")


def test_every_bundled_graph_lints_clean():
    from graph_looper import available, load

    for name in available():
        assert lint(load(name)) == [], f"{name} has lint warnings"
