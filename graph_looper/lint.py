"""Warnings about graphs that are valid but probably wrong.

`Graph.validate()` proves a graph *can* run — every node reachable, every gate
branch covered. Lint asks a different question: does every node earn its place?

The costly mistake it catches is a node that runs, bills you, and has its output
silently dropped because no downstream prompt ever references it. That is easy
to do by hand and easier still for a model writing a graph for you.
"""

from __future__ import annotations

from dataclasses import dataclass

from graph_looper import template
from graph_looper.spec import Graph, Node


@dataclass(frozen=True)
class LintWarning:
    """One suspicious thing. Never fatal on its own."""

    node: str | None
    code: str
    message: str

    def __str__(self) -> str:
        where = f"{self.node}: " if self.node else ""
        return f"{where}{self.message} [{self.code}]"


def node_templates(node: Node) -> list[str]:
    """Every template string a node renders."""
    return [t for t in (node.prompt, node.system, node.source) if t]


def referenced(node: Node) -> list[str]:
    """Every `{{ path }}` a node's templates mention."""
    paths: list[str] = []
    for text in node_templates(node):
        for path in template.referenced_paths(text):
            if path not in paths:
                paths.append(path)
    return paths


def lint(graph: Graph) -> list[LintWarning]:
    """Every warning for *graph*, in node order."""
    warnings: list[LintWarning] = []
    references = {node.id: referenced(node) for node in graph.nodes}

    warnings.extend(_unreferenced_outputs(graph, references))
    warnings.extend(_dangling_inputs(graph, references))
    warnings.extend(_unused_vars(graph, references))
    warnings.extend(_unread_state(graph, references))
    return warnings


def _consumers_of(graph: Graph, node_id: str) -> set[str]:
    return {edge.target for edge in graph.outgoing(node_id)}


def _unreferenced_outputs(
    graph: Graph, references: dict[str, list[str]]
) -> list[LintWarning]:
    """A node whose text nobody reads is a node you are paying for twice."""
    warnings: list[LintWarning] = []
    for node in graph.nodes:
        # A gate's output *is* its routing decision, and an output node's text
        # is the run's result. Neither needs to be quoted anywhere.
        if node.type in ("gate", "output", "input"):
            continue
        # Writing to state is a use, even if no prompt reads it back yet.
        if node.writes_state:
            continue

        wanted = {f"inputs.{node.id}", f"results.{node.id}", f"data.{node.id}"}
        consumers = _consumers_of(graph, node.id)
        used = False
        for other_id, paths in references.items():
            # `data.a.verdict` reads node `a` just as much as `data.a` does, so
            # compare on the first two segments rather than the whole path.
            if any(".".join(p.split(".")[:2]) in wanted for p in paths):
                used = True
                break
            # `{{ input }}` splices in every incoming message, so a direct
            # consumer using it does read this node.
            if other_id in consumers and "input" in paths:
                used = True
                break
        if not used:
            warnings.append(
                LintWarning(
                    node=node.id,
                    code="unreferenced-output",
                    message=(
                        "runs, but no prompt reads its result — reference it with "
                        f"{{{{ inputs.{node.id} }}}} or {{{{ input }}}} downstream, "
                        "or remove the node"
                    ),
                )
            )
    return warnings


def _dangling_inputs(
    graph: Graph, references: dict[str, list[str]]
) -> list[LintWarning]:
    """`{{ inputs.x }}` only fills when x is an incoming edge — otherwise it is
    permanently blank and the prompt silently loses a section."""
    warnings: list[LintWarning] = []
    for node in graph.nodes:
        sources = {edge.source for edge in graph.incoming(node.id)}
        for path in references[node.id]:
            if not path.startswith(("inputs.", "data.")):
                continue
            named = path.split(".", 1)[1].split(".")[0]
            if named in sources:
                continue
            if not graph.has(named):
                warnings.append(
                    LintWarning(
                        node=node.id,
                        code="unknown-node-reference",
                        message=(
                            f"{{{{ {path} }}}} names {named!r}, which is not a node "
                            "in this graph — it will always render empty"
                        ),
                    )
                )
            else:
                warnings.append(
                    LintWarning(
                        node=node.id,
                        code="dangling-input",
                        message=(
                            f"{{{{ {path} }}}} reads {named!r}, but no edge runs "
                            f"{named} → {node.id}; use {{{{ results.{named} }}}} "
                            "if you meant its latest value from anywhere in the run"
                        ),
                    )
                )
    return warnings


def _unused_vars(graph: Graph, references: dict[str, list[str]]) -> list[LintWarning]:
    used = {
        path.split(".", 1)[1]
        for paths in references.values()
        for path in paths
        if path.startswith("vars.")
    }
    return [
        LintWarning(
            node=None,
            code="unused-var",
            message=f"vars.{name} is declared but no prompt reads it",
        )
        for name in sorted(set(graph.vars) - used)
    ]


def _unread_state(graph: Graph, references: dict[str, list[str]]) -> list[LintWarning]:
    """State written but never read back does nothing for future runs."""
    read = {
        path.split(".", 1)[1].split(".")[0]
        for paths in references.values()
        for path in paths
        if path.startswith("state.")
    }
    warnings: list[LintWarning] = []
    for node in graph.nodes:
        if node.writes_state and node.writes_state not in read:
            warnings.append(
                LintWarning(
                    node=node.id,
                    code="write-only-state",
                    message=(
                        f"writes state.{node.writes_state}, but no prompt reads it "
                        "back — later runs will not benefit"
                    ),
                )
            )
    return warnings
