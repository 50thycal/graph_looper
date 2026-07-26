from __future__ import annotations

import pytest

from graph_looper.providers import MockProvider
from graph_looper.runtime import Runner
from graph_looper.spec import load_graph_str


def run(graph, responses=None, task="the task", **kwargs):
    provider = MockProvider(responses=responses or {})
    result = Runner(graph, provider).run(task, **kwargs)
    return result, provider


def test_linear_run_reaches_the_output(linear_graph):
    result, provider = run(linear_graph, {"step": ["stepped"]})
    assert result.ok
    assert result.final_node == "done"
    assert result.output == "stepped"
    assert provider.calls_for("step")[0].prompt == "do the task"


def test_fan_out_runs_concurrently_and_join_all_waits(fanout_graph):
    result, _ = run(fanout_graph, {"a": ["A"], "b": ["B"]})
    assert result.ok
    assert result.output == "A + B"
    # a and b fired in the same tick — the trace records both before merge.
    order = [e.node for e in result.trace.events if e.kind == "node_end"]
    assert order.index("merge") > max(order.index("a"), order.index("b"))


def test_loop_repeats_until_the_gate_passes(loop_graph):
    result, provider = run(
        loop_graph,
        {
            "worker": ["first", "second"],
            "check": [
                {"choice": "fail", "reason": "not yet"},
                {"choice": "pass", "reason": "good"},
            ],
        },
    )
    assert result.ok
    assert len(provider.calls_for("worker")) == 2
    assert result.results["worker"].visit == 2


def test_gate_is_forced_once_its_visit_budget_is_spent(loop_graph):
    result, provider = run(
        loop_graph, {"check": [{"choice": "fail", "reason": "never happy"}]}
    )
    assert result.ok, result.error
    check = result.results["check"]
    assert check.forced is True
    assert check.label == "pass"
    # The budget is 2, so the model was asked twice and forced on the third.
    assert len(provider.calls_for("check")) == 2


def test_resident_agent_remembers_its_own_attempts(loop_graph):
    _, provider = run(
        loop_graph,
        {
            "worker": ["first", "second"],
            "check": [
                {"choice": "fail", "reason": "no"},
                {"choice": "pass", "reason": "yes"},
            ],
        },
    )
    second_call = provider.calls_for("worker")[1]
    roles = [m["role"] for m in second_call.messages]
    assert roles == ["user", "assistant", "user"]
    assert second_call.messages[1]["content"] == "first"


def test_ephemeral_agent_starts_fresh_every_visit(loop_graph):
    graph = load_graph_str(
        loop_graph.to_yaml().replace("mode: resident", "mode: ephemeral")
    )
    _, provider = run(
        graph,
        {
            "worker": ["first", "second"],
            "check": [
                {"choice": "fail", "reason": "no"},
                {"choice": "pass", "reason": "yes"},
            ],
        },
    )
    assert all(len(c.messages) == 1 for c in provider.calls_for("worker"))


def test_feedback_reaches_the_next_attempt(loop_graph):
    _, provider = run(
        loop_graph,
        {
            "worker": ["first", "second"],
            "check": [
                {"choice": "fail", "reason": "needs a title"},
                {"choice": "pass", "reason": "fine"},
            ],
        },
    )
    retry_prompt = provider.calls_for("worker")[1].prompt
    assert "attempt 2" in retry_prompt
    assert "needs a title" in retry_prompt


def test_gate_choice_outside_its_choices_is_an_error(loop_graph):
    result, _ = run(loop_graph, {"check": [{"choice": "maybe", "reason": "?"}]})
    assert not result.ok
    assert "not one of" in result.error


def test_join_any_fires_on_the_first_arrival():
    graph = load_graph_str(
        """
        name: any
        nodes:
          - id: start
            type: input
          - id: slow
            type: agent
            prompt: slow
          - id: fast
            type: agent
            prompt: fast
          - id: sink
            type: output
            join: any
            prompt: "{{ input }}"
        edges:
          - from: start
            to: slow
          - from: start
            to: fast
          - from: slow
            to: sink
          - from: fast
            to: sink
        """
    )
    result, _ = run(graph, {"slow": ["S"], "fast": ["F"]})
    assert result.ok
    # Both arrive in the same tick here, so `any` sees both; the point is that
    # it does not deadlock waiting for a branch that will never fire.
    assert set(result.output.split()) & {"S", "F"}


def test_label_from_routes_on_a_structured_field():
    graph = load_graph_str(
        """
        name: routed
        nodes:
          - id: start
            type: input
          - id: classify
            type: agent
            label_from: data.kind
            output_schema:
              type: object
              properties:
                kind:
                  type: string
            prompt: classify
          - id: left
            type: agent
            prompt: left
          - id: right
            type: agent
            prompt: right
          - id: done
            type: output
            join: any
            prompt: "{{ input }}"
        edges:
          - from: start
            to: classify
          - from: classify
            to: left
            when: a
          - from: classify
            to: right
            when: b
          - from: left
            to: done
          - from: right
            to: done
        """
    )
    result, provider = run(graph, {"classify": [{"kind": "b"}], "right": ["went right"]})
    assert result.ok
    assert result.output == "went right"
    assert provider.calls_for("left") == []


def test_stall_is_reported_with_what_it_was_waiting_for():
    graph = load_graph_str(
        """
        name: stall
        nodes:
          - id: start
            type: input
          - id: g
            type: gate
            choices: [go, stop]
            on_exhausted: go
            prompt: pick
          - id: never
            type: agent
            prompt: never
          - id: join_here
            type: agent
            prompt: "{{ input }}"
          - id: done
            type: output
        edges:
          - from: start
            to: g
          - from: g
            to: join_here
            when: go
          - from: g
            to: never
            when: stop
          - from: never
            to: join_here
          - from: join_here
            to: done
        """
    )
    result, _ = run(graph, {"g": [{"choice": "go", "reason": "go"}]})
    assert not result.ok
    assert "stalled" in result.error
    assert "join_here still waiting on never" in result.error


RUNAWAY = """
name: runaway
limits:
  {limit}
nodes:
  - id: start
    type: input
  - id: a
    type: agent
    join: any
    label_from: data.state
    output_schema:
      type: object
      properties:
        state:
          type: string
    prompt: a
  - id: b
    type: agent
    prompt: b
  - id: done
    type: output
edges:
  - from: start
    to: a
  - from: a
    to: b
    when: loop
  - from: b
    to: a
  - from: a
    to: done
    when: stop
"""
# The agent never emits the label that would leave the loop.
NEVER_STOPS = {"a": [{"state": "loop"}]}


def test_step_limit_stops_a_runaway_graph():
    graph = load_graph_str(RUNAWAY.format(limit="max_steps: 6"))
    result, _ = run(graph, NEVER_STOPS)
    assert not result.ok
    assert "step limit" in result.error


def test_node_visit_cap_stops_a_tight_loop():
    graph = load_graph_str(RUNAWAY.format(limit="max_node_visits: 3"))
    result, _ = run(graph, NEVER_STOPS)
    assert not result.ok
    assert "cap 3" in result.error


@pytest.mark.parametrize(
    "op,expected",
    [("concat", "A\n\nB"), ("first", "A"), ("last", "B")],
)
def test_transform_ops(op, expected):
    graph = load_graph_str(
        f"""
        name: t
        nodes:
          - id: start
            type: input
          - id: a
            type: agent
            prompt: a
          - id: b
            type: agent
            prompt: b
          - id: t
            type: transform
            op: {op}
          - id: done
            type: output
            prompt: "{{{{ inputs.t }}}}"
        edges:
          - from: start
            to: a
          - from: start
            to: b
          - from: a
            to: t
          - from: b
            to: t
          - from: t
            to: done
        """
    )
    result, _ = run(graph, {"a": ["A"], "b": ["B"]})
    assert result.output == expected


def test_variables_are_available_to_prompts(linear_graph):
    graph = load_graph_str(
        linear_graph.to_yaml().replace('do {{ task }}', 'do {{ task }} for {{ vars.who }}')
    )
    _, provider = run(graph, {"step": ["ok"]}, variables={"who": "Cal"})
    assert provider.calls_for("step")[0].prompt == "do the task for Cal"


def test_trace_records_the_path_taken(loop_graph):
    result, _ = run(
        loop_graph,
        {
            "check": [
                {"choice": "fail", "reason": "no"},
                {"choice": "pass", "reason": "yes"},
            ]
        },
    )
    assert ("check", "worker") in result.trace.traversed_edges()
    assert ("check", "done") in result.trace.traversed_edges()
    assert result.trace.events[0].kind == "run_start"
    assert result.trace.events[-1].kind == "run_end"
    assert result.trace.events[0].at == 0.0


def test_result_serializes_to_json_safe_dict(linear_graph):
    result, _ = run(linear_graph, {"step": ["ok"]})
    import json

    payload = json.dumps(result.to_dict())
    assert '"ok": true' in payload
