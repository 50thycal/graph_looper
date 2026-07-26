from __future__ import annotations

import pytest

from graph_looper.template import (
    TemplateError,
    referenced_paths,
    render,
    resolve,
    stringify,
)

CONTEXT = {
    "task": "ship it",
    "iteration": 2,
    "inputs": {"planner": "the plan"},
    "data": {"gate": {"choice": "pass"}},
    "vars": {"tone": "dry"},
}


def test_substitutes_dotted_paths():
    assert render("{{ task }} / {{ inputs.planner }}", CONTEXT) == "ship it / the plan"


def test_ignores_single_braces_so_json_examples_survive():
    template = 'Return {"choice": "pass"} for {{ task }}'
    assert render(template, CONTEXT) == 'Return {"choice": "pass"} for ship it'


def test_missing_paths_render_empty():
    assert render("[{{ results.nothing }}]", CONTEXT) == "[]"


def test_strict_mode_raises_on_missing_paths():
    with pytest.raises(TemplateError):
        render("{{ nope }}", CONTEXT, strict=True)


def test_numbers_and_nested_values():
    assert render("{{ iteration }}", CONTEXT) == "2"
    assert render("{{ data.gate.choice }}", CONTEXT) == "pass"


def test_whitespace_inside_braces_is_optional():
    assert render("{{task}}{{  task  }}", CONTEXT) == "ship itship it"


def test_referenced_paths_are_deduped_in_order():
    assert referenced_paths("{{ b }} {{ a }} {{ b }}") == ["b", "a"]


def test_resolve_returns_none_for_unknown_paths():
    assert resolve("inputs.absent", CONTEXT) is None
    assert resolve("task.deeper", CONTEXT) is None


def test_stringify_renders_structures_as_json():
    assert stringify({"a": 1}) == '{\n  "a": 1\n}'
    assert stringify(None) == ""
    assert stringify(True) == "true"
