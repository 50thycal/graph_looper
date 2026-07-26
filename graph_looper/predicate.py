"""Rule matching for gates that route without a model call.

Most branches in a real workflow are not judgement calls — "does this ticket
mention a refund" does not need a frontier model. A predicate gate decides from
local rules at zero tokens and zero latency, and falls through to the model only
when nothing matches and no default is set.

Deliberately not an expression language: five matchers, first match wins. Rules
are data you can read at a glance, not code you have to trust.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Decision:
    """The outcome of running a rule set."""

    choice: str | None
    reason: str
    rule_index: int | None = None

    @property
    def matched(self) -> bool:
        return self.choice is not None


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def match_rule(rule: dict[str, Any], text: str) -> tuple[bool, str]:
    """Test one rule against *text*, returning whether it hit and why."""
    haystack = text or ""

    if "always" in rule:
        return bool(rule["always"]), "catch-all rule"

    if "not_empty" in rule:
        wants = bool(rule["not_empty"])
        is_empty = not haystack.strip()
        hit = (not is_empty) if wants else is_empty
        return hit, "input is empty" if is_empty else "input is non-empty"

    if "equals" in rule:
        expected = str(rule["equals"])
        hit = haystack.strip().casefold() == expected.strip().casefold()
        return hit, f"equals {expected!r}" if hit else f"does not equal {expected!r}"

    if "contains" in rule:
        needles = _as_list(rule["contains"])
        folded = haystack.casefold()
        for needle in needles:
            if needle.casefold() in folded:
                return True, f"contains {needle!r}"
        return False, f"contains none of {needles}"

    if "matches" in rule:
        pattern = str(rule["matches"])
        found = re.search(pattern, haystack, re.IGNORECASE | re.MULTILINE)
        if found:
            return True, f"matches /{pattern}/ at {found.start()}"
        return False, f"does not match /{pattern}/"

    return False, "rule has no matcher"


def evaluate(rules: Sequence[dict[str, Any]], text: str) -> Decision:
    """Run *rules* in order and return the first match."""
    for index, rule in enumerate(rules):
        hit, why = match_rule(rule, text)
        if hit:
            return Decision(
                choice=str(rule.get("choice")),
                reason=f"rule {index + 1}: {why}",
                rule_index=index,
            )
    return Decision(choice=None, reason=f"no rule matched ({len(rules)} tried)")
