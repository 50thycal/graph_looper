from __future__ import annotations

import pytest

from graph_looper.cli import bundled_graphs
from graph_looper.spec import Graph, GraphError, load_graph, load_graph_str

MINIMAL = """
name: t
nodes:
  - id: start
    type: input
  - id: done
    type: output
edges:
  - from: start
    to: done
"""


def bad(document: str) -> str:
    with pytest.raises(GraphError) as excinfo:
        load_graph_str(document)
    return str(excinfo.value)


def test_minimal_graph_parses():
    graph = load_graph_str(MINIMAL)
    assert graph.name == "t"
    assert [n.id for n in graph.nodes] == ["start", "done"]
    assert graph.outgoing("start")[0].target == "done"


def test_every_bundled_graph_is_valid():
    names = bundled_graphs()
    assert "reviewer-loop" in names
    for path in names.values():
        load_graph(path)


def test_duplicate_ids_rejected():
    assert "duplicate node id" in bad(
        MINIMAL.replace("  - id: done\n    type: output", "  - id: start\n    type: output")
    )


def test_unknown_edge_endpoint_rejected():
    assert "unknown target" in bad(MINIMAL + "  - from: start\n    to: nowhere\n")


def test_graph_needs_an_output():
    assert "at least one node of type 'output'" in bad(
        """
        name: t
        nodes:
          - id: start
            type: input
          - id: a
            type: agent
            prompt: hi
        edges:
          - from: start
            to: a
        """
    )


def test_gate_needs_on_exhausted_and_full_coverage():
    message = bad(
        """
        name: t
        nodes:
          - id: start
            type: input
          - id: g
            type: gate
            choices: ["ship", "hold"]
            prompt: pick
          - id: done
            type: output
        edges:
          - from: start
            to: g
          - from: g
            to: done
            when: ship
        """
    )
    assert "on_exhausted" in message
    assert "gate choice(s) hold have no outgoing edge" in message


def test_unquoted_yes_no_choices_survive_yaml_booleans():
    """`choices: [yes, no]` is [True, False] to a YAML parser — fold it back."""
    graph = load_graph_str(
        """
        name: t
        nodes:
          - id: start
            type: input
          - id: g
            type: gate
            choices: [yes, no]
            on_exhausted: yes
            prompt: pick
          - id: done
            type: output
        edges:
          - from: start
            to: g
          - from: g
            to: done
            when: yes
          - from: g
            to: done
            when: no
        """
    )
    gate = graph.node("g")
    assert gate.choices == ["true", "false"]
    assert gate.on_exhausted == "true"
    assert {e.when for e in graph.outgoing("g")} == {"true", "false"}


def test_gate_edge_condition_must_be_a_choice():
    assert "are not gate choices" in bad(
        """
        name: t
        nodes:
          - id: start
            type: input
          - id: g
            type: gate
            choices: [a, b]
            on_exhausted: a
            prompt: pick
          - id: done
            type: output
        edges:
          - from: start
            to: g
          - from: g
            to: done
            when: a
          - from: g
            to: done
            when: b
          - from: g
            to: done
            when: c
        """
    )


def test_unreachable_and_dead_end_nodes_rejected():
    message = bad(
        """
        name: t
        nodes:
          - id: start
            type: input
          - id: orphan
            type: agent
            prompt: hi
          - id: done
            type: output
        edges:
          - from: start
            to: done
          - from: orphan
            to: orphan
        """
    )
    assert "unreachable from any input" in message


def test_agent_without_prompt_rejected():
    assert "need a 'prompt'" in bad(
        MINIMAL.replace(
            "  - id: done\n    type: output",
            "  - id: a\n    type: agent\n  - id: done\n    type: output",
        ).replace("  - from: start\n    to: done", "  - from: start\n    to: a\n  - from: a\n    to: done")
    )


def test_unknown_keys_are_reported():
    assert "unknown key(s) colour" in bad(
        MINIMAL.replace("  - id: start\n    type: input", "  - id: start\n    type: input\n    colour: red")
    )


def test_input_cannot_have_incoming_edges():
    assert "input nodes cannot have incoming edges" in bad(
        MINIMAL + "  - from: done\n    to: start\n"
    )


def test_roundtrip_through_dict(loop_graph: Graph):
    again = Graph.from_dict(loop_graph.to_dict())
    assert again.to_dict() == loop_graph.to_dict()
    assert "max_visits: 2" in loop_graph.to_yaml()


def test_visit_cap_defaults(loop_graph: Graph):
    assert loop_graph.max_visits_for(loop_graph.node("check")) == 2
