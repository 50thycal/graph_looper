"""The public entry points.

Everything another project needs is here, and the shortest useful call is one
line::

    from graph_looper import run

    result = run("reviewer-loop", "Write the Q3 explainer.")
    print(result.output)

`load()` accepts whatever you already have — a `Graph`, a path, a bundled or
registered name, a plain dict, or YAML text — so a caller never has to care
which form their graph is in.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from graph_looper import catalog, render as _render
from graph_looper.lint import LintWarning, lint as _lint
from graph_looper.providers import Provider
from graph_looper.runtime import Runner, RunResult, Trace, TraceEvent
from graph_looper.spec import Graph, GraphError

#: What `load()` will take.
GraphSource = "Graph | str | Path | dict[str, Any]"


def _looks_like_document(text: str) -> bool:
    """Distinguish inline YAML from a graph name or path."""
    return "\n" in text and "nodes:" in text


def load(
    source: Any,
    *,
    search_paths: Sequence[str | Path] | None = None,
) -> Graph:
    """Return a validated `Graph` from whatever *source* is.

    Args:
        source: a `Graph` (returned unchanged), a `Path` or path-like string, a
            bundled or registered graph name, a `dict` in the graph schema, or
            YAML text.
        search_paths: extra directories to look in when *source* is a name.
            `GRAPHLOOPER_PATH` and the bundled graphs are always searched too.

    Raises:
        GraphError: if the graph cannot be found, parsed, or validated.
    """
    if isinstance(source, Graph):
        return source
    if isinstance(source, dict):
        return Graph.from_dict(source)
    if isinstance(source, Path):
        return Graph.from_file(source)
    if isinstance(source, str):
        if _looks_like_document(source):
            return Graph.from_yaml(source)
        return Graph.from_file(catalog.resolve(source, search_paths))
    raise GraphError(
        "load() takes a Graph, a path, a graph name, a dict, or YAML text; "
        f"got {type(source).__name__}"
    )


def run(
    graph: Any,
    task: str,
    *,
    provider: Provider | None = None,
    state: Any = None,
    namespace: str | None = None,
    variables: dict[str, Any] | None = None,
    on_event: Callable[[TraceEvent], None] | None = None,
    search_paths: Sequence[str | Path] | None = None,
) -> RunResult:
    """Execute a graph and return everything the run produced.

    Args:
        graph: anything `load()` accepts.
        task: the run's input, available to prompts as `{{ task }}`.
        provider: where model calls go. Defaults to `AnthropicProvider()`; pass
            `MockProvider()` to exercise a graph with no network.
        state: persistence between runs — a `StateStore`, a path, `True` for the
            default `.graphloop/state.json`, or `None` for no memory.
        namespace: the key state is filed under. Defaults to the graph's name.
        variables: values for `{{ vars.* }}`, merged over the graph's own.
        on_event: called with each `TraceEvent` as the run proceeds.
        search_paths: extra directories to resolve a graph name against.

    Returns:
        A `RunResult`. Check `.ok`, read `.output`, and call
        `.raise_for_status()` if you would rather have an exception.
    """
    return asyncio.run(
        arun(
            graph,
            task,
            provider=provider,
            state=state,
            namespace=namespace,
            variables=variables,
            on_event=on_event,
            search_paths=search_paths,
        )
    )


async def arun(
    graph: Any,
    task: str,
    *,
    provider: Provider | None = None,
    state: Any = None,
    namespace: str | None = None,
    variables: dict[str, Any] | None = None,
    on_event: Callable[[TraceEvent], None] | None = None,
    search_paths: Sequence[str | Path] | None = None,
) -> RunResult:
    """The async form of `run()`, for callers already inside an event loop."""
    runner = Runner(
        load(graph, search_paths=search_paths),
        provider,
        state=state,
        namespace=namespace,
        on_event=on_event,
    )
    return await runner.arun(task, variables=variables)


def validate(
    graph: Any, *, search_paths: Sequence[str | Path] | None = None
) -> Graph:
    """Load a graph and confirm it is runnable. Raises `GraphError` if not."""
    return load(graph, search_paths=search_paths)


def lint(
    graph: Any, *, search_paths: Sequence[str | Path] | None = None
) -> list[LintWarning]:
    """Non-fatal warnings: nodes nothing reads, references that never fill."""
    return _lint(load(graph, search_paths=search_paths))


def to_mermaid(
    graph: Any,
    *,
    trace: Trace | None = None,
    direction: str = "LR",
    legend: bool = True,
    search_paths: Sequence[str | Path] | None = None,
) -> str:
    """Render a graph as a Mermaid `flowchart`.

    Pass a `RunResult.trace` to highlight the path a run actually took.
    """
    return _render.to_mermaid(
        load(graph, search_paths=search_paths),
        trace=trace,
        direction=direction,
        legend=legend,
    )


def available(
    search_paths: Iterable[str | Path] | None = None,
) -> dict[str, Path]:
    """Every graph findable by name, mapped to its file."""
    return catalog.available(search_paths)
