"""A deliberately small `{{ dotted.path }}` template renderer.

Prompts are full of braces (JSON examples, code), so single braces are left
alone and only `{{ ... }}` is substituted. Unknown paths render as the empty
string by default — a node that fires before one of its optional upstreams has
produced anything is normal in a loop, not an error.
"""

from __future__ import annotations

import re
from typing import Any

_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_-]+)*)\s*\}\}")


class TemplateError(KeyError):
    """Raised in strict mode when a template references an unknown path."""


def referenced_paths(template: str) -> list[str]:
    """Every `{{ path }}` referenced by a template, in order of first use."""
    seen: list[str] = []
    for match in _PATTERN.finditer(template or ""):
        path = match.group(1)
        if path not in seen:
            seen.append(path)
    return seen


def resolve(path: str, context: dict[str, Any]) -> Any:
    """Walk a dotted path through nested mappings/objects. Returns None if absent."""
    current: Any = context
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, (list, tuple)):
            if not part.isdigit() or int(part) >= len(current):
                return None
            current = current[int(part)]
        else:
            current = getattr(current, part, None)
            if current is None:
                return None
    return current


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    import json

    return json.dumps(value, indent=2, ensure_ascii=False)


def render(template: str, context: dict[str, Any], *, strict: bool = False) -> str:
    """Substitute every `{{ path }}` in *template* using *context*."""
    if not template:
        return ""

    def _sub(match: re.Match[str]) -> str:
        path = match.group(1)
        value = resolve(path, context)
        if value is None and strict:
            raise TemplateError(f"template references unknown path {path!r}")
        return stringify(value)

    return _PATTERN.sub(_sub, template)
