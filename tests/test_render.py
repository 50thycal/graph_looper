from __future__ import annotations

from graph_looper import render
from graph_looper.providers import MockProvider
from graph_looper.runtime import Runner


def test_mermaid_has_a_line_per_node_and_edge(loop_graph):
    diagram = render.to_mermaid(loop_graph)
    assert diagram.startswith("flowchart LR")
    assert 'worker["worker"]' in diagram
    assert 'check{"check?"}' in diagram  # gates get a question mark
    assert "start --> worker" in diagram
    assert 'check -- "fail" --> worker' in diagram


def test_resident_and_ephemeral_get_different_classes(loop_graph):
    diagram = render.to_mermaid(loop_graph)
    assert "class worker resident;" in diagram
    assert "legend_ephemeral" in diagram


def test_quotes_in_labels_are_escaped(loop_graph):
    loop_graph.node("worker").title = 'Worker "the builder"'
    assert 'worker["Worker #quot;the builder#quot;"]' in render.to_mermaid(loop_graph)


def test_trace_overlay_marks_visits_and_dims_untaken_paths(loop_graph):
    provider = MockProvider(
        responses={
            "check": [
                {"choice": "fail", "reason": "no"},
                {"choice": "pass", "reason": "yes"},
            ]
        }
    )
    result = Runner(loop_graph, provider).run("task")
    diagram = render.to_mermaid(loop_graph, trace=result.trace)
    assert 'worker["worker ×2"]' in diagram
    assert "stroke:#2f8f4e" in diagram


def test_markdown_wraps_the_diagram(loop_graph):
    text = render.to_markdown(loop_graph)
    assert "```mermaid" in text
    assert text.strip().endswith("```")


def test_find_cycles_reports_the_feedback_loop(loop_graph):
    cycles = render.find_cycles(loop_graph)
    assert cycles == [["worker", "check"]] or cycles == [["check", "worker"]]


def test_find_cycles_is_empty_for_a_dag(fanout_graph):
    assert render.find_cycles(fanout_graph) == []


def test_text_summary_lists_loops_and_gate_budgets(loop_graph):
    text = render.to_text(loop_graph)
    assert "pass/fail" in text
    assert "budget 2 → pass" in text
    assert "loops (1)" in text


def test_trace_summary_is_readable(loop_graph):
    provider = MockProvider(responses={"check": [{"choice": "pass", "reason": "ok"}]})
    result = Runner(loop_graph, provider).run("task")
    summary = render.trace_summary(result.trace)
    assert "▶ loop" in summary
    assert "✓ check → pass" in summary
    assert "■ done" in summary
