"""Author a graph spec from a description — or from a photo of one you drew.

This closes the loop the whole project is about: sketch the workflow on paper,
photograph it, and get back a spec that actually runs. The generated graph is
validated, and validation errors are fed back to the model for another pass.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from graph_looper.providers import LLMRequest, Provider
from graph_looper.spec import Graph, GraphError

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

DSL_REFERENCE = """\
You design executable agent-workflow graphs. Output one graph in the schema provided.

NODE TYPES
  input      Entry point. Receives the run's task. Exactly one, usually.
  agent      An LLM call. `prompt` is required. `mode` is either:
               ephemeral - fresh context every visit (reviewers, one-shot steps)
               resident  - keeps its conversation across visits, so on a retry it
                           remembers its own earlier attempt (workers, planners)
  gate       An LLM call that must pick one of `choices`; edges route on the pick.
             Requires `choices` (>= 2) and `on_exhausted` (the choice forced once
             `max_visits` is spent, which is what stops a loop spinning forever).
  transform  No model call. `op` is concat | first | last | template | json.
  output     Terminal node. Its text is the run's result.

EDGES
  Every edge is {from, to}. An edge out of a gate carries `when: <choice>`, and
  every choice of every gate must have at least one outgoing edge.
  Cycles are allowed and expected — a "fail" edge pointing back at an earlier
  node is how revision loops are built.

JOINS
  `join: all` (default) - the node waits for a message on every incoming edge.
                          Use for fan-in: a synthesiser waiting on N reviewers.
  `join: any`           - fires on whichever message arrives first. Use for any
                          node that sits inside a loop and is fed both by its
                          normal upstream and by a feedback edge; `all` would
                          deadlock there.

PROMPTS
  Prompts are templates. `{{ task }}` is the run input. `{{ inputs.<node_id> }}`
  is the text from that upstream node for this firing. `{{ input }}` is every
  incoming message joined under headings. `{{ iteration }}` is the visit count.
  `{{ vars.<key> }}` reads a run variable. Unknown paths render empty, which is
  fine for a node whose feedback edge has not fired yet.

RULES
  - Node ids are lowercase snake_case and unique.
  - Every node must be reachable from an input and able to reach an output.
  - Non-input nodes need at least one incoming edge; non-output nodes need at
    least one outgoing edge.
  - Write real, specific prompts. Each one should state the role, the input it
    is looking at, and the shape of the output expected. No placeholders.
  - Prefer few strong nodes over many trivial ones.

Leave a field as "" (or 0, or []) when it does not apply to that node type.
"""

GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "kebab-case identifier"},
        "description": {"type": "string", "description": "One sentence."},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["input", "agent", "gate", "transform", "output"],
                    },
                    "title": {"type": "string", "description": "Diagram label, or ''."},
                    "mode": {"type": "string", "enum": ["ephemeral", "resident", ""]},
                    "system": {"type": "string"},
                    "prompt": {"type": "string"},
                    "choices": {"type": "array", "items": {"type": "string"}},
                    "max_visits": {"type": "integer", "description": "0 for default."},
                    "on_exhausted": {"type": "string"},
                    "op": {
                        "type": "string",
                        "enum": ["concat", "first", "last", "template", "json", ""],
                    },
                    "join": {"type": "string", "enum": ["all", "any", ""]},
                },
                "required": [
                    "id",
                    "type",
                    "title",
                    "mode",
                    "system",
                    "prompt",
                    "choices",
                    "max_visits",
                    "on_exhausted",
                    "op",
                    "join",
                ],
                "additionalProperties": False,
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "when": {"type": "string", "description": "Gate choice, or ''."},
                    "label": {"type": "string", "description": "Diagram label, or ''."},
                },
                "required": ["from", "to", "when", "label"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["name", "description", "nodes", "edges"],
    "additionalProperties": False,
}

_SENTINELS = ("", 0, [], None)


def clean_spec(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop the placeholder values the schema forces the model to emit."""
    out: dict[str, Any] = {
        "name": raw.get("name") or "generated-graph",
        "nodes": [],
        "edges": [],
    }
    if raw.get("description"):
        out["description"] = raw["description"]

    for node in raw.get("nodes") or []:
        cleaned = {k: v for k, v in node.items() if v not in _SENTINELS}
        cleaned.setdefault("id", node.get("id", ""))
        cleaned.setdefault("type", node.get("type", "agent"))
        # `mode` only means something on agents; `op` only on transforms.
        if cleaned.get("type") != "agent":
            cleaned.pop("mode", None)
        if cleaned.get("type") != "transform":
            cleaned.pop("op", None)
        if cleaned.get("type") != "gate":
            cleaned.pop("choices", None)
            cleaned.pop("on_exhausted", None)
        out["nodes"].append(cleaned)

    for edge in raw.get("edges") or []:
        out["edges"].append({k: v for k, v in edge.items() if v not in _SENTINELS})
    return out


def image_block(path: str | Path) -> dict[str, Any]:
    """A base64 image content block for a sketch of a graph."""
    path = Path(path)
    media_type = _MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise GraphError(
            f"{path}: unsupported image type; use one of "
            f"{', '.join(sorted(_MEDIA_TYPES))}"
        )
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


async def author_graph(
    provider: Provider,
    *,
    description: str = "",
    images: list[str | Path] | None = None,
    model: str = "claude-opus-5",
    effort: str = "high",
    max_tokens: int = 16000,
    attempts: int = 3,
) -> tuple[Graph, list[str]]:
    """Generate and validate a graph. Returns the graph and a log of the attempts."""
    if not description and not images:
        raise GraphError("authoring needs a description, an image, or both")

    content: list[dict[str, Any]] = []
    for image in images or []:
        content.append(image_block(image))
    ask = description.strip() or (
        "Read the workflow drawn in the image(s) and turn it into a runnable graph. "
        "Preserve every node, branch, and feedback loop shown, including the "
        "resident-vs-ephemeral distinction if the drawing marks one."
    )
    content.append({"type": "text", "text": ask})

    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    log: list[str] = []

    for attempt in range(1, attempts + 1):
        response = await provider.complete(
            LLMRequest(
                node_id="author",
                messages=messages,
                system=DSL_REFERENCE,
                model=model,
                max_tokens=max_tokens,
                effort=effort,
                schema=GRAPH_SCHEMA,
            )
        )
        raw = response.data
        if raw is None:
            try:
                raw = json.loads(response.text)
            except json.JSONDecodeError:
                raw = None
        if raw is None:
            log.append(f"attempt {attempt}: model did not return a JSON graph")
            messages.append({"role": "assistant", "content": response.text})
            messages.append(
                {
                    "role": "user",
                    "content": "That was not valid JSON. Return the graph object only.",
                }
            )
            continue

        try:
            graph = Graph.from_dict(clean_spec(raw))
        except GraphError as exc:
            log.append(f"attempt {attempt}: {exc}")
            if attempt == attempts:
                raise GraphError(
                    "could not produce a valid graph after "
                    f"{attempts} attempts.\n" + "\n".join(log)
                ) from None
            messages.append(
                {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)}
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That graph failed validation:\n{exc}\n\n"
                        "Fix exactly those problems and return the whole graph again."
                    ),
                }
            )
            continue

        log.append(f"attempt {attempt}: valid")
        return graph, log

    raise GraphError("could not produce a valid graph.\n" + "\n".join(log))
