"""The live provider, exercised against a fake client.

There is no API key in CI, so these pin the request we *would* send and the way
we read the response back — the parts most likely to rot.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from graph_looper.providers import (
    AnthropicProvider,
    LLMRequest,
    ProviderError,
    choice_schema,
    normalize_schema,
)


def block(kind: str, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(type=kind, text=text, thinking=text)


class FakeStream:
    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self.message


class FakeClient:
    """Captures the kwargs a real call would have received."""

    def __init__(self, message):
        self.message = message
        self.captured: dict = {}
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.captured = kwargs
        return FakeStream(self.message)


def message(
    content=None, *, stop_reason="end_turn", model="claude-opus-5", tokens=(10, 20)
):
    return SimpleNamespace(
        content=content if content is not None else [block("text", "hello")],
        stop_reason=stop_reason,
        stop_details=None,
        model=model,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
    )


def complete(request: LLMRequest, msg=None) -> tuple:
    client = FakeClient(msg or message())
    response = asyncio.run(AnthropicProvider(client=client).complete(request))
    return response, client.captured


def test_request_shape():
    response, sent = complete(
        LLMRequest(
            node_id="n",
            messages=[{"role": "user", "content": "hi"}],
            system="be terse",
            model="claude-opus-5",
            max_tokens=4321,
            effort="medium",
        )
    )
    assert sent["model"] == "claude-opus-5"
    assert sent["max_tokens"] == 4321
    assert sent["system"] == "be terse"
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "medium"}
    assert "format" not in sent["output_config"]
    assert response.text == "hello"
    assert (response.input_tokens, response.output_tokens) == (10, 20)


def test_system_is_omitted_when_absent():
    _, sent = complete(
        LLMRequest(node_id="n", messages=[{"role": "user", "content": "hi"}])
    )
    assert "system" not in sent


def test_schema_becomes_structured_output_and_is_normalized():
    request = LLMRequest(
        node_id="n",
        messages=[{"role": "user", "content": "hi"}],
        schema={"type": "object", "properties": {"kind": {"type": "string"}}},
    )
    response, sent = complete(request, message([block("text", '{"kind": "bug"}')]))
    fmt = sent["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert fmt["schema"]["required"] == ["kind"]
    assert response.data == {"kind": "bug"}


def test_disabling_thinking_caps_effort_at_high():
    """`thinking: disabled` above `high` effort is a 400 — step it down instead."""
    _, sent = complete(
        LLMRequest(
            node_id="n",
            messages=[{"role": "user", "content": "hi"}],
            thinking=False,
            effort="max",
        )
    )
    assert sent["thinking"] == {"type": "disabled"}
    assert sent["output_config"]["effort"] == "high"


def test_disabling_thinking_leaves_low_effort_alone():
    _, sent = complete(
        LLMRequest(
            node_id="n",
            messages=[{"role": "user", "content": "hi"}],
            thinking=False,
            effort="low",
        )
    )
    assert sent["output_config"]["effort"] == "low"


def test_thinking_blocks_are_not_mistaken_for_the_answer():
    response, _ = complete(
        LLMRequest(node_id="n", messages=[{"role": "user", "content": "hi"}]),
        message([block("thinking", "pondering"), block("text", "the answer")]),
    )
    assert response.text == "the answer"


def test_refusal_is_reported_rather_than_returned_as_content():
    msg = message([], stop_reason="refusal")
    msg.stop_details = SimpleNamespace(category="cyber", explanation="no")
    with pytest.raises(ProviderError) as excinfo:
        complete(
            LLMRequest(node_id="gate", messages=[{"role": "user", "content": "hi"}]), msg
        )
    assert "declined" in str(excinfo.value)
    assert "cyber" in str(excinfo.value)


def test_unparseable_structured_output_names_the_node():
    with pytest.raises(ProviderError) as excinfo:
        complete(
            LLMRequest(
                node_id="classify",
                messages=[{"role": "user", "content": "hi"}],
                schema={"type": "object", "properties": {}},
            ),
            message([block("text", "not json")]),
        )
    assert "classify" in str(excinfo.value)


def test_api_failures_carry_the_node_id():
    class Boom:
        def __init__(self):
            self.messages = SimpleNamespace(stream=self._stream)

        def _stream(self, **kwargs):
            raise RuntimeError("connection reset")

    provider = AnthropicProvider(client=Boom())
    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(
            provider.complete(
                LLMRequest(node_id="worker", messages=[{"role": "user", "content": "x"}])
            )
        )
    assert "worker" in str(excinfo.value)
    assert "connection reset" in str(excinfo.value)


def test_normalize_schema_recurses_into_nested_objects_and_arrays():
    schema = normalize_schema(
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                }
            },
        }
    )
    inner = schema["properties"]["items"]["items"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["name"]


def test_normalize_schema_keeps_an_explicit_required_list():
    schema = normalize_schema(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
        }
    )
    assert schema["required"] == ["a"]


def test_choice_schema_enumerates_the_gate_branches():
    schema = choice_schema(["pass", "fail"])
    assert schema["properties"]["choice"]["enum"] == ["pass", "fail"]
    assert schema["required"] == ["choice", "reason"]


def test_prompt_reads_the_last_user_turn():
    request = LLMRequest(
        node_id="n",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ],
    )
    assert request.prompt == "second"
