"""Render a graph as Mermaid (optionally overlaid with a run trace) or as text."""

from __future__ import annotations

from typing import Any

from graph_looper.runtime import Trace
from graph_looper.spec import Graph, Node

_SHAPES = {
    "input": ("([", "])"),
    "agent": ("[", "]"),
    "gate": ("{", "}"),
    "transform": ("[/", "/]"),
    "output": ("((", "))"),
}

_STYLES = """  classDef io fill:#1f2933,stroke:#8899a6,color:#f5f7fa;
  classDef ephemeral fill:#ffffff,stroke:#5b6b7b,color:#1f2933;
  classDef resident fill:#e6f5ea,stroke:#2f8f4e,color:#12351f,stroke-width:2px;
  classDef gate fill:#3b2f10,stroke:#b8860b,color:#f7efd8;
  classDef transform fill:#eef2ff,stroke:#5566cc,color:#1a2050;
  classDef idle opacity:0.45,stroke-dasharray:4 3;"""


def _escape(text: str) -> str:
    return (
        text.replace('"', "#quot;")
        .replace("\n", " ")
        .replace("{", "(")
        .replace("}", ")")
    )


def _node_class(node: Node) -> str:
    if node.type in ("input", "output"):
        return "io"
    if node.type == "gate":
        return "gate"
    if node.type == "transform":
        return "transform"
    return "resident" if node.mode == "resident" else "ephemeral"


def _node_caption(node: Node, visits: int | None) -> str:
    caption = node.label
    if node.type == "gate":
        caption = caption if caption.endswith("?") else f"{caption}?"
    if visits:
        caption = f"{caption} ×{visits}" if visits > 1 else caption
    return caption


def to_mermaid(
    graph: Graph,
    *,
    trace: Trace | None = None,
    direction: str = "LR",
    legend: bool = True,
) -> str:
    """A Mermaid `flowchart`. With a trace, the path actually taken is highlighted."""
    visit_counts: dict[str, int] = {}
    traversed: set[tuple[str, str]] = set()
    if trace is not None:
        for event in trace.events:
            if event.kind == "node_end" and event.node:
                visit_counts[event.node] = visit_counts.get(event.node, 0) + 1
        traversed = set(trace.traversed_edges())

    lines = [f"flowchart {direction}"]
    for node in graph.nodes:
        open_shape, close_shape = _SHAPES.get(node.type, ("[", "]"))
        caption = _escape(_node_caption(node, visit_counts.get(node.id)))
        lines.append(f'  {node.id}{open_shape}"{caption}"{close_shape}')

    for edge in graph.edges:
        label = edge.display_label()
        arrow = f'-- "{_escape(label)}" -->' if label else "-->"
        lines.append(f"  {edge.source} {arrow} {edge.target}")

    lines.append("")
    lines.append(_STYLES)
    for node in graph.nodes:
        lines.append(f"  class {node.id} {_node_class(node)};")

    if trace is not None:
        idle = [n.id for n in graph.nodes if n.id not in visit_counts]
        if idle:
            lines.append(f"  class {','.join(idle)} idle;")
        for index, edge in enumerate(graph.edges):
            if (edge.source, edge.target) in traversed:
                lines.append(
                    f"  linkStyle {index} stroke:#2f8f4e,stroke-width:2.5px;"
                )
            else:
                lines.append(f"  linkStyle {index} stroke-dasharray:4 3,opacity:0.35;")

    if legend:
        lines.append("")
        lines.append("  subgraph legend [ ]")
        lines.append('    legend_resident["resident agent"]')
        lines.append('    legend_ephemeral["ephemeral worker"]')
        lines.append("  end")
        lines.append("  class legend_resident resident;")
        lines.append("  class legend_ephemeral ephemeral;")
        lines.append("  style legend fill:none,stroke:none;")

    return "\n".join(lines) + "\n"


def to_markdown(graph: Graph, *, trace: Trace | None = None) -> str:
    """A Mermaid diagram wrapped in a fenced block, ready to paste anywhere."""
    header = f"# {graph.name}\n"
    if graph.description:
        header += f"\n{graph.description}\n"
    body = to_mermaid(graph, trace=trace)
    return f"{header}\n```mermaid\n{body}```\n"


def to_text(graph: Graph) -> str:
    """A compact textual summary — what `validate` prints."""
    lines = [f"{graph.name}"]
    if graph.description:
        lines.append(f"  {graph.description}")
    lines.append("")
    lines.append(f"  nodes ({len(graph.nodes)}):")
    for node in graph.nodes:
        bits: list[str] = [node.type]
        if node.type == "agent":
            bits.append(node.mode)
        if node.type == "gate":
            bits.append("/".join(node.choices))
            bits.append(f"budget {graph.max_visits_for(node)} → {node.on_exhausted}")
        if node.type == "transform":
            bits.append(node.op)
        if node.join != "all" and graph.incoming(node.id):
            bits.append(f"join {node.join}")
        lines.append(f"    {node.id:<22} {' · '.join(bits)}")
    lines.append("")
    lines.append(f"  edges ({len(graph.edges)}):")
    for edge in graph.edges:
        label = edge.display_label()
        suffix = f"   [{label}]" if label else ""
        lines.append(f"    {edge.source} → {edge.target}{suffix}")
    cycles = find_cycles(graph)
    if cycles:
        lines.append("")
        lines.append(f"  loops ({len(cycles)}):")
        for cycle in cycles:
            lines.append("    " + " → ".join(cycle + [cycle[0]]))
    return "\n".join(lines)


def find_cycles(graph: Graph) -> list[list[str]]:
    """Every distinct simple cycle, so `validate` can show what loops exist."""
    cycles: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    path: list[str] = []
    on_path: set[str] = set()

    def walk(node_id: str) -> None:
        path.append(node_id)
        on_path.add(node_id)
        for edge in graph.outgoing(node_id):
            if edge.target in on_path:
                cycle = path[path.index(edge.target) :]
                key = _canonical(cycle)
                if key not in seen:
                    seen.add(key)
                    cycles.append(list(cycle))
            elif len(path) < len(graph.nodes):
                walk(edge.target)
        path.pop()
        on_path.discard(node_id)

    for node in graph.nodes:
        walk(node.id)
    return cycles


def _canonical(cycle: list[str]) -> tuple[str, ...]:
    pivot = cycle.index(min(cycle))
    return tuple(cycle[pivot:] + cycle[:pivot])


def trace_summary(trace: Trace) -> str:
    """A human-readable replay of a run."""
    lines: list[str] = []
    for event in trace.events:
        detail: dict[str, Any] = event.detail
        stamp = f"{event.at:6.2f}s"
        if event.kind == "run_start":
            lines.append(f"{stamp}  ▶ {detail.get('graph')}: {detail.get('task')}")
        elif event.kind == "node_start":
            lines.append(
                f"{stamp}  · {event.node} starting (visit {detail.get('visit')})"
            )
        elif event.kind == "node_end":
            label = detail.get("label")
            tag = f" → {label}" if label else ""
            forced = " (forced)" if detail.get("forced") else ""
            lines.append(
                f"{stamp}  ✓ {event.node}{tag}{forced}: {detail.get('preview', '')}"
            )
        elif event.kind == "edge":
            when = detail.get("when")
            lines.append(
                f"{stamp}    ↳ {detail['from']} → {detail['to']}"
                + (f" [{when}]" if when else "")
            )
        elif event.kind == "dropped":
            lines.append(f"{stamp}  ! {event.node}: {detail.get('message')}")
        elif event.kind == "error":
            lines.append(f"{stamp}  ✗ {detail.get('message')}")
        elif event.kind == "run_end":
            status = "done" if detail.get("ok") else "failed"
            lines.append(
                f"{stamp}  ■ {status} in {detail.get('steps')} steps"
            )
    return "\n".join(lines)
