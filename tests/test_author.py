from __future__ import annotations

import asyncio

import pytest

from graph_looper.author import GRAPH_SCHEMA, author_graph, clean_spec, image_block
from graph_looper.providers import MockProvider
from graph_looper.spec import GraphError

VALID = {
    "name": "made-up",
    "description": "A generated loop.",
    "nodes": [
        {
            "id": "start",
            "type": "input",
            "title": "Task",
            "mode": "",
            "system": "",
            "prompt": "",
            "choices": [],
            "max_visits": 0,
            "on_exhausted": "",
            "op": "",
            "join": "",
        },
        {
            "id": "work",
            "type": "agent",
            "title": "Work",
            "mode": "resident",
            "system": "",
            "prompt": "do {{ task }}",
            "choices": [],
            "max_visits": 0,
            "on_exhausted": "",
            "op": "",
            "join": "",
        },
        {
            "id": "done",
            "type": "output",
            "title": "Done",
            "mode": "",
            "system": "",
            "prompt": "",
            "choices": [],
            "max_visits": 0,
            "on_exhausted": "",
            "op": "",
            "join": "",
        },
    ],
    "edges": [
        {"from": "start", "to": "work", "when": "", "label": ""},
        {"from": "work", "to": "done", "when": "", "label": ""},
    ],
}


def test_clean_spec_strips_placeholder_values():
    cleaned = clean_spec(VALID)
    start = next(n for n in cleaned["nodes"] if n["id"] == "start")
    assert "mode" not in start  # not an agent
    assert "max_visits" not in start
    assert cleaned["edges"][0] == {"from": "start", "to": "work"}
    work = next(n for n in cleaned["nodes"] if n["id"] == "work")
    assert work["mode"] == "resident"


def test_schema_is_strict_enough_for_structured_outputs():
    def check(schema):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
            assert set(schema["required"]) == set(schema["properties"])
            for child in schema["properties"].values():
                check(child)
        if schema.get("type") == "array":
            check(schema["items"])

    check(GRAPH_SCHEMA)


def test_author_graph_returns_a_validated_graph():
    provider = MockProvider(responses={"author": [VALID]})
    graph, log = asyncio.run(author_graph(provider, description="a loop"))
    assert graph.name == "made-up"
    assert graph.node("work").mode == "resident"
    assert log == ["attempt 1: valid"]


def test_author_graph_feeds_validation_errors_back_and_retries():
    broken = {**VALID, "edges": [{"from": "start", "to": "work", "when": "", "label": ""}]}
    provider = MockProvider(responses={"author": [broken, VALID]})
    graph, log = asyncio.run(author_graph(provider, description="a loop"))
    assert graph.name == "made-up"
    assert len(log) == 2 and log[1] == "attempt 2: valid"
    # The retry carried the validation error back to the model.
    retry = provider.calls_for("author")[1]
    assert "failed validation" in retry.prompt


def test_author_graph_gives_up_with_the_reasons():
    broken = {**VALID, "nodes": []}
    provider = MockProvider(responses={"author": [broken]})
    with pytest.raises(GraphError) as excinfo:
        asyncio.run(author_graph(provider, description="x", attempts=2))
    assert "after 2 attempts" in str(excinfo.value)


def test_author_graph_requires_some_input():
    with pytest.raises(GraphError):
        asyncio.run(author_graph(MockProvider()))


def test_image_block_rejects_unsupported_types(tmp_path):
    path = tmp_path / "sketch.bmp"
    path.write_bytes(b"nope")
    with pytest.raises(GraphError):
        image_block(path)


def test_image_block_encodes_png(tmp_path):
    path = tmp_path / "sketch.png"
    path.write_bytes(b"\x89PNG fake")
    block = image_block(path)
    assert block["source"]["media_type"] == "image/png"
    assert block["source"]["type"] == "base64"


def test_images_are_sent_before_the_instruction(tmp_path):
    path = tmp_path / "sketch.png"
    path.write_bytes(b"\x89PNG fake")
    provider = MockProvider(responses={"author": [VALID]})
    asyncio.run(author_graph(provider, images=[path]))
    content = provider.calls[0].messages[0]["content"]
    assert content[0]["type"] == "image"
    assert content[-1]["type"] == "text"
