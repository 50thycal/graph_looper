from __future__ import annotations

import pytest

from graph_looper import MockProvider, run
from graph_looper.predicate import evaluate, match_rule
from graph_looper.spec import GraphError, load_graph_str

ROUTED = """
name: routed
nodes:
  - id: start
    type: input
  - id: pick
    type: gate
    mode: predicate
    choices: [urgent, normal]
    on_exhausted: normal
    source: "{{ task }}"
    rules:
      - contains: [asap, urgent, "now"]
        choice: urgent
    default: normal
  - id: fast
    type: agent
    prompt: fast
  - id: slow
    type: agent
    prompt: slow
  - id: done
    type: output
    join: any
    prompt: "{{ input }}"
edges:
  - from: start
    to: pick
  - from: pick
    to: fast
    when: urgent
  - from: pick
    to: slow
    when: normal
  - from: fast
    to: done
  - from: slow
    to: done
"""


@pytest.mark.parametrize(
    "rule,text,expected",
    [
        ({"contains": ["cat"]}, "the CAT sat", True),
        ({"contains": ["cat"]}, "the dog sat", False),
        ({"contains": "cat"}, "a cat", True),  # bare string, not a list
        ({"matches": r"\d{3}"}, "code 404 here", True),
        ({"matches": r"^start"}, "no", False),
        ({"matches": "ERROR"}, "an error occurred", True),  # case-insensitive
        ({"equals": "yes"}, "  YES  ", True),
        ({"equals": "yes"}, "yes please", False),
        ({"not_empty": True}, "  ", False),
        ({"not_empty": True}, "x", True),
        ({"not_empty": False}, "", True),
        ({"always": True}, "anything", True),
        ({"always": False}, "anything", False),
        ({}, "anything", False),
    ],
)
def test_matchers(rule, text, expected):
    hit, _ = match_rule(rule, text)
    assert hit is expected


def test_matchers_tolerate_empty_text():
    assert match_rule({"contains": ["x"]}, "")[0] is False


def test_first_matching_rule_wins():
    rules = [
        {"contains": ["b"], "choice": "second"},
        {"contains": ["a"], "choice": "third"},
    ]
    decision = evaluate(rules, "a and b")
    assert decision.choice == "second"
    assert decision.rule_index == 0


def test_no_match_is_reported_not_guessed():
    decision = evaluate([{"contains": ["z"], "choice": "x"}], "abc")
    assert not decision.matched
    assert decision.choice is None
    assert "no rule matched" in decision.reason


def test_reason_explains_the_decision_for_the_trace():
    decision = evaluate([{"contains": ["invoice"], "choice": "billing"}], "my invoice")
    assert "contains 'invoice'" in decision.reason


# -- as a gate in a real run -----------------------------------------------


def test_predicate_gate_routes_without_calling_the_model():
    provider = MockProvider({"fast": ["handled"]})
    result = run(ROUTED, "need this ASAP", provider=provider)
    assert result.ok
    assert result.results["pick"].label == "urgent"
    assert result.results["pick"].via == "predicate"
    assert result.usage_by_node["pick"]["model_calls"] == 0
    assert provider.calls_for("pick") == []


def test_default_catches_what_the_rules_miss():
    result = run(ROUTED, "whenever you get a chance", provider=MockProvider({"slow": ["ok"]}))
    assert result.results["pick"].label == "normal"
    assert result.results["pick"].via == "predicate"


def test_without_a_default_an_unmatched_gate_escalates_to_the_model():
    graph = ROUTED.replace("    default: normal\n", "").replace(
        "    rules:", "    prompt: decide this one properly\n    rules:"
    )
    provider = MockProvider({"pick": [{"choice": "urgent", "reason": "judged"}]})
    result = run(graph, "ambiguous wording", provider=provider)
    assert result.results["pick"].label == "urgent"
    assert result.results["pick"].via == "model"
    assert len(provider.calls_for("pick")) == 1


def test_source_defaults_to_the_incoming_messages():
    graph = ROUTED.replace('    source: "{{ task }}"\n', "")
    result = run(graph, "urgent thing", provider=MockProvider({"fast": ["ok"]}))
    # The input node passes the task through, so `{{ input }}` sees it.
    assert result.results["pick"].label == "urgent"


def test_visit_budget_still_forces_a_predicate_gate():
    graph = ROUTED.replace("    on_exhausted: normal", "    on_exhausted: normal\n    max_visits: 0")
    with pytest.raises(GraphError):
        load_graph_str(graph)


def test_a_predicate_gate_needs_rules():
    graph = ROUTED.replace("    rules:\n      - contains: [asap, urgent, \"now\"]\n        choice: urgent\n", "")
    with pytest.raises(GraphError) as excinfo:
        load_graph_str(graph)
    assert "needs 'rules'" in str(excinfo.value)


def test_rules_require_mode_predicate():
    with pytest.raises(GraphError) as excinfo:
        load_graph_str(ROUTED.replace("    mode: predicate\n", ""))
    assert "mode: predicate" in str(excinfo.value)


def test_a_rule_choice_must_be_a_gate_choice():
    with pytest.raises(GraphError) as excinfo:
        load_graph_str(ROUTED.replace("        choice: urgent", "        choice: nonsense"))
    assert "not one of" in str(excinfo.value)


def test_a_rule_needs_exactly_one_matcher():
    with pytest.raises(GraphError) as excinfo:
        load_graph_str(
            ROUTED.replace(
                "      - contains: [asap, urgent, \"now\"]",
                "      - contains: [asap]\n        equals: urgent",
            )
        )
    assert "exactly one of" in str(excinfo.value)


def test_a_bad_regex_fails_at_load_not_mid_run():
    with pytest.raises(GraphError) as excinfo:
        load_graph_str(
            ROUTED.replace("      - contains: [asap, urgent, \"now\"]", '      - matches: "([unclosed"')
        )
    assert "not a valid regex" in str(excinfo.value)


def test_default_must_be_a_gate_choice():
    with pytest.raises(GraphError) as excinfo:
        load_graph_str(ROUTED.replace("    default: normal", "    default: nonsense"))
    assert "default 'nonsense'" in str(excinfo.value)


def test_a_predicate_gate_with_a_default_needs_no_prompt():
    """The whole point: this node never reaches the model, so it needs no prompt."""
    graph = load_graph_str(ROUTED)
    assert graph.node("pick").prompt is None
    assert graph.node("pick").calls_model() is False
