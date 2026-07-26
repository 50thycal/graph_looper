"""Model providers.

`AnthropicProvider` makes real Claude calls. `MockProvider` replays scripted
answers so graphs can be exercised — loops, gates, joins and all — without a
network call or an API key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

DEFAULT_MODEL = "claude-opus-5"


class ProviderError(RuntimeError):
    """A model call failed or came back unusable."""


@dataclass
class LLMRequest:
    node_id: str
    messages: list[dict[str, Any]]
    system: str | None = None
    model: str = DEFAULT_MODEL
    max_tokens: int = 8000
    effort: str = "high"
    thinking: bool = True
    schema: dict[str, Any] | None = None

    @property
    def prompt(self) -> str:
        """The text of the final user turn — what this call is actually asking."""
        for message in reversed(self.messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "\n".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
        return ""


@dataclass
class LLMResponse:
    text: str
    data: dict[str, Any] | None = None
    model: str = ""
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Provider(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...


def normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a hand-written schema acceptable to structured outputs.

    Structured outputs require `additionalProperties: false` and an explicit
    `required` list on every object. Graph authors forget both, constantly.
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if out.get("type") == "object":
        properties = out.get("properties") or {}
        out["properties"] = {k: normalize_schema(v) for k, v in properties.items()}
        out.setdefault("required", list(properties))
        out["additionalProperties"] = False
    elif out.get("type") == "array" and isinstance(out.get("items"), dict):
        out["items"] = normalize_schema(out["items"])
    return out


def choice_schema(choices: Sequence[str]) -> dict[str, Any]:
    """The structured-output schema a gate uses to pick a branch."""
    return {
        "type": "object",
        "properties": {
            "choice": {
                "type": "string",
                "enum": list(choices),
                "description": "Which branch to take.",
            },
            "reason": {
                "type": "string",
                "description": "One or two sentences on why, for the trace.",
            },
        },
        "required": ["choice", "reason"],
        "additionalProperties": False,
    }


class AnthropicProvider:
    """Calls Claude. Streams, so large `max_tokens` doesn't trip HTTP timeouts."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ProviderError(
                    "the 'anthropic' package is required for live runs; "
                    "install it or use --dry-run"
                ) from exc
            # Credentials resolve from the environment: ANTHROPIC_API_KEY,
            # ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile.
            self._client = AsyncAnthropic()
        return self._client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        effort = request.effort
        thinking: dict[str, Any] = {"type": "adaptive"}
        if not request.thinking:
            # Disabling thinking is only accepted at effort `high` or below.
            thinking = {"type": "disabled"}
            if effort in ("xhigh", "max"):
                effort = "high"

        output_config: dict[str, Any] = {"effort": effort}
        if request.schema:
            output_config["format"] = {
                "type": "json_schema",
                "schema": normalize_schema(request.schema),
            }

        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": request.messages,
            "thinking": thinking,
            "output_config": output_config,
        }
        if request.system:
            kwargs["system"] = request.system

        try:
            async with self.client.messages.stream(**kwargs) as stream:
                message = await stream.get_final_message()
        except Exception as exc:  # noqa: BLE001 - surfaced with node context
            raise ProviderError(f"{request.node_id}: model call failed: {exc}") from exc

        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ProviderError(
                f"{request.node_id}: the model declined this request"
                + (f" (category: {category})" if category else "")
            )

        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        ).strip()

        data: dict[str, Any] | None = None
        if request.schema:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"{request.node_id}: expected JSON matching the node's schema, "
                    f"got {text[:200]!r}"
                ) from exc
            data = parsed if isinstance(parsed, dict) else {"value": parsed}

        usage = getattr(message, "usage", None)
        return LLMResponse(
            text=text,
            data=data,
            model=getattr(message, "model", request.model),
            stop_reason=getattr(message, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


@dataclass
class MockProvider:
    """Replays scripted answers, keyed by node id.

    Each entry is a list consumed one call at a time; once a node's script runs
    out the final entry repeats, so a loop whose last scripted verdict is "pass"
    always terminates.
    """

    responses: dict[str, list[Any]] = field(default_factory=dict)
    fallback: Callable[[LLMRequest], Any] | None = None
    calls: list[LLMRequest] = field(default_factory=list)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        scripted = self._next(request)
        return self._to_response(scripted, request)

    def calls_for(self, node_id: str) -> list[LLMRequest]:
        return [c for c in self.calls if c.node_id == node_id]

    def _next(self, request: LLMRequest) -> Any:
        script = self.responses.get(request.node_id)
        if script:
            used = sum(1 for c in self.calls[:-1] if c.node_id == request.node_id)
            return script[min(used, len(script) - 1)]
        if self.fallback is not None:
            return self.fallback(request)
        if request.schema:
            enum = (
                (request.schema.get("properties") or {})
                .get("choice", {})
                .get("enum")
            )
            if enum:
                return {"choice": enum[0], "reason": "mock provider default"}
            return {}
        return f"[mock:{request.node_id}]"

    @staticmethod
    def _to_response(scripted: Any, request: LLMRequest) -> LLMResponse:
        if isinstance(scripted, LLMResponse):
            return scripted
        if isinstance(scripted, dict):
            return LLMResponse(
                text=json.dumps(scripted, ensure_ascii=False),
                data=scripted,
                model="mock",
                output_tokens=len(json.dumps(scripted)) // 4,
            )
        text = str(scripted)
        data = None
        if request.schema:
            try:
                loaded = json.loads(text)
                data = loaded if isinstance(loaded, dict) else {"value": loaded}
            except json.JSONDecodeError:
                data = None
        return LLMResponse(
            text=text, data=data, model="mock", output_tokens=len(text) // 4
        )
