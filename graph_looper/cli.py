"""Command line interface: `graphloop`."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

from graph_looper import render
from graph_looper.providers import AnthropicProvider, MockProvider, Provider
from graph_looper.runtime import Runner, Trace, TraceEvent
from graph_looper.spec import GraphError, load_graph

BUNDLED = Path(__file__).parent / "graphs"


# -- helpers ---------------------------------------------------------------


def bundled_graphs() -> dict[str, Path]:
    if not BUNDLED.is_dir():
        return {}
    return {p.stem: p for p in sorted(BUNDLED.glob("*.yaml"))}


def resolve_graph(reference: str) -> Path:
    """Accept a path, or the bare name of a bundled example."""
    path = Path(reference)
    if path.exists():
        return path
    bundled = bundled_graphs()
    if reference in bundled:
        return bundled[reference]
    known = ", ".join(bundled) or "none installed"
    raise GraphError(f"no graph at {reference!r}; bundled graphs: {known}")


def parse_vars(pairs: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise GraphError(f"--var expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value
    return out


def load_trace(path: str | Path) -> Trace:
    raw = json.loads(Path(path).read_text())
    events_raw = raw.get("trace", raw) if isinstance(raw, dict) else raw
    trace = Trace()
    for item in events_raw:
        detail = {
            k: v for k, v in item.items() if k not in ("seq", "at", "kind", "node")
        }
        trace.events.append(
            TraceEvent(
                seq=item.get("seq", len(trace.events)),
                at=item.get("at", 0.0),
                kind=item.get("kind", ""),
                node=item.get("node"),
                detail=detail,
            )
        )
    return trace


def build_provider(args: argparse.Namespace) -> Provider:
    if getattr(args, "mock", None):
        script = json.loads(Path(args.mock).read_text())
        return MockProvider(responses=script)
    if getattr(args, "dry_run", False):
        return MockProvider()
    return AnthropicProvider()


def write_out(path: str | None, text: str, *, what: str) -> None:
    if path:
        Path(path).write_text(text)
        print(f"wrote {what} to {path}", file=sys.stderr)
    else:
        sys.stdout.write(text)


# -- commands --------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    graph = load_graph(resolve_graph(args.graph))
    print(f"✓ {graph.name} is valid ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")
    if args.verbose:
        print()
        print(render.to_text(graph))
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    graph = load_graph(resolve_graph(args.graph))
    trace = load_trace(args.trace) if args.trace else None
    if args.markdown or (args.output or "").endswith(".md"):
        text = render.to_markdown(graph, trace=trace)
    else:
        text = render.to_mermaid(
            graph, trace=trace, direction=args.direction, legend=not args.no_legend
        )
    write_out(args.output, text, what="diagram")
    return 0


def cmd_examples(args: argparse.Namespace) -> int:
    graphs = bundled_graphs()
    if not graphs:
        print("no bundled graphs found", file=sys.stderr)
        return 1
    for name, path in graphs.items():
        try:
            description = load_graph(path).description or ""
        except GraphError:
            description = "(fails validation)"
        print(f"{name:<22} {description}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    graph = load_graph(resolve_graph(args.graph))
    task = args.input
    if args.input_file:
        task = Path(args.input_file).read_text()
    if not task:
        raise GraphError("give the run a task with --input or --input-file")

    quiet = args.quiet or args.json
    provider = build_provider(args)

    def on_event(event: TraceEvent) -> None:
        if quiet:
            return
        if event.kind == "node_start":
            print(f"  · {event.node} (visit {event.detail.get('visit')})", file=sys.stderr)
        elif event.kind == "node_end":
            label = event.detail.get("label")
            tag = f" → {label}" if label else ""
            forced = " [forced]" if event.detail.get("forced") else ""
            print(f"  ✓ {event.node}{tag}{forced}", file=sys.stderr)
        elif event.kind == "error":
            print(f"  ✗ {event.detail.get('message')}", file=sys.stderr)

    if not quiet:
        print(f"▶ {graph.name}", file=sys.stderr)

    runner = Runner(graph, provider, on_event=on_event)
    result = runner.run(task, variables=parse_vars(args.var))

    if args.trace:
        Path(args.trace).write_text(json.dumps(result.to_dict(), indent=2))
        if not quiet:
            print(f"trace written to {args.trace}", file=sys.stderr)

    if args.viz:
        Path(args.viz).write_text(render.to_markdown(graph, trace=result.trace))
        if not quiet:
            print(f"diagram written to {args.viz}", file=sys.stderr)

    if args.json:
        print(json.dumps(result.to_dict(include_trace=args.trace is None), indent=2))
    else:
        if not quiet:
            print(
                f"■ {'done' if result.ok else 'failed'} · {result.steps} steps · "
                f"{result.seconds:.1f}s · "
                f"{result.input_tokens + result.output_tokens} tokens",
                file=sys.stderr,
            )
            print(file=sys.stderr)
        if result.ok:
            print(result.output)
        else:
            print(result.error, file=sys.stderr)

    return 0 if result.ok else 1


def cmd_author(args: argparse.Namespace) -> int:
    from graph_looper.author import author_graph

    provider = build_provider(args)
    graph, log = asyncio.run(
        author_graph(
            provider,
            description=args.description or "",
            images=args.image,
            model=args.model,
            effort=args.effort,
            attempts=args.attempts,
        )
    )
    for line in log:
        print(f"  {line}", file=sys.stderr)
    write_out(args.output, graph.to_yaml(), what="graph")
    if args.output:
        print(
            f"try it: graphloop run {args.output} --input '...' --dry-run",
            file=sys.stderr,
        )
    return 0


# -- parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphloop",
        description="Draw an agent workflow as a graph; run it as a real multi-agent loop.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute a graph")
    run.add_argument("graph", help="path to a graph file, or a bundled graph name")
    run.add_argument("-i", "--input", help="the task to run")
    run.add_argument("--input-file", help="read the task from a file")
    run.add_argument(
        "--var", action="append", default=[], metavar="K=V", help="template variable"
    )
    run.add_argument("--trace", help="write the full run trace as JSON here")
    run.add_argument("--viz", help="write a Mermaid diagram of the path taken here")
    run.add_argument("--json", action="store_true", help="print the result as JSON")
    run.add_argument("--quiet", "-q", action="store_true", help="suppress progress")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="use the mock provider — exercises routing with no model calls",
    )
    run.add_argument("--mock", help="JSON file of scripted responses per node id")
    run.set_defaults(func=cmd_run)

    viz = sub.add_parser("viz", help="render a graph as Mermaid")
    viz.add_argument("graph")
    viz.add_argument("-o", "--output", help="write here instead of stdout")
    viz.add_argument("--trace", help="overlay the path taken by a saved run")
    viz.add_argument("--markdown", action="store_true", help="wrap in a fenced block")
    viz.add_argument("--direction", default="LR", choices=["LR", "TB", "RL", "BT"])
    viz.add_argument("--no-legend", action="store_true")
    viz.set_defaults(func=cmd_viz)

    validate = sub.add_parser("validate", help="check a graph and describe it")
    validate.add_argument("graph")
    validate.add_argument("-v", "--verbose", action="store_true")
    validate.set_defaults(func=cmd_validate)

    author = sub.add_parser(
        "author", help="generate a graph from a description or a photo of a sketch"
    )
    author.add_argument("description", nargs="?", help="what the workflow should do")
    author.add_argument(
        "--image",
        action="append",
        default=[],
        help="a picture of the graph (repeatable)",
    )
    author.add_argument("-o", "--output", help="write the graph here")
    author.add_argument("--model", default="claude-opus-5")
    author.add_argument(
        "--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"]
    )
    author.add_argument("--attempts", type=int, default=3)
    author.add_argument("--mock", help=argparse.SUPPRESS)
    author.set_defaults(func=cmd_author)

    examples = sub.add_parser("examples", help="list the bundled graphs")
    examples.set_defaults(func=cmd_examples)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except GraphError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
