"""Graph specification: parsing and validation.

A graph is a set of nodes joined by directed edges. Cycles are allowed and are
the point — feedback loops are what make these workflows interesting. Loops are
bounded by per-node visit budgets rather than by acyclicity.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from graph_looper.lint import LintWarning

import yaml

NODE_TYPES = ("input", "agent", "gate", "transform", "output")
JOIN_MODES = ("all", "any")
AGENT_MODES = ("ephemeral", "resident")
GATE_MODES = ("llm", "predicate")
TRANSFORM_OPS = ("concat", "first", "last", "template", "json")
MATCHERS = ("contains", "matches", "equals", "not_empty", "always")

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 8000
DEFAULT_EFFORT = "high"
DEFAULT_MAX_VISITS = 3
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_SECONDS = 1800
DEFAULT_MAX_NODE_VISITS = 25
DEFAULT_MAX_PARALLEL = 8


class GraphError(ValueError):
    """Raised when a graph specification is malformed."""


def _as_label(value: Any) -> Any:
    """Normalise a routing label written in YAML.

    Unquoted `yes`/`no`/`on`/`off` are booleans to a YAML parser, which would
    otherwise leave a gate's choices and its edge conditions as `True`/`False`
    and fail to match anything readable. Fold them back to text so
    `choices: [yes, no]` behaves the way it looks.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return value


@dataclass
class Node:
    id: str
    type: str = "agent"
    # Display
    title: str | None = None
    # Agent / gate
    mode: str | None = None  # agents: ephemeral | resident. gates: llm | predicate
    system: str | None = None
    prompt: str | None = None
    model: str | None = None
    max_tokens: int | None = None
    effort: str | None = None
    thinking: bool = True
    output_schema: dict[str, Any] | None = None
    # Gate
    choices: list[str] = field(default_factory=list)
    max_visits: int | None = None
    on_exhausted: str | None = None
    # Gate, predicate mode
    rules: list[dict[str, Any]] = field(default_factory=list)
    default: str | None = None
    source: str | None = None
    # Transform
    op: str = "concat"
    separator: str = "\n\n"
    # Routing
    join: str = "all"
    label_from: str | None = None
    # Persistence
    writes_state: str | None = None
    state_append: bool = False
    state_limit: int | None = None
    # Context budget
    keep_last: int | None = None
    max_input_chars: int | None = None

    @property
    def label(self) -> str:
        return self.title or self.id.replace("_", " ")

    @property
    def agent_mode(self) -> str:
        """`ephemeral` (fresh context each visit) or `resident` (keeps its own)."""
        return self.mode or "ephemeral"

    @property
    def gate_mode(self) -> str:
        """`llm` (ask the model) or `predicate` (decide with local rules)."""
        return self.mode or "llm"

    def is_llm(self) -> bool:
        """Node types that *can* reach the model."""
        return self.type in ("agent", "gate")

    def calls_model(self) -> bool:
        """Whether it actually will.

        A predicate gate with a `default` never does — that is the whole point
        of it, and it means such a node needs no prompt.
        """
        if self.type == "agent":
            return True
        if self.type == "gate":
            return self.gate_mode == "llm" or not self.default
        return False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Node":
        if not isinstance(raw, dict):
            raise GraphError(f"node must be a mapping, got {type(raw).__name__}")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise GraphError(
                f"node {raw.get('id', '<no id>')!r}: unknown key(s) "
                f"{', '.join(sorted(unknown))}"
            )
        node_id = raw.get("id")
        if not node_id or not isinstance(node_id, str):
            raise GraphError("every node needs a string 'id'")
        data = dict(raw)
        data["id"] = node_id
        if "choices" in data:
            if not isinstance(data["choices"], list):
                raise GraphError(f"node {node_id!r}: 'choices' must be a list")
            data["choices"] = [_as_label(c) for c in data["choices"]]
        if "on_exhausted" in data:
            data["on_exhausted"] = _as_label(data["on_exhausted"])
        if "default" in data:
            data["default"] = _as_label(data["default"])
        if "rules" in data:
            if not isinstance(data["rules"], list):
                raise GraphError(f"node {node_id!r}: 'rules' must be a list")
            data["rules"] = [
                {**r, "choice": _as_label(r["choice"])}
                if isinstance(r, dict) and "choice" in r
                else r
                for r in data["rules"]
            ]
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        defaults = Node(id="_")
        out: dict[str, Any] = {"id": self.id, "type": self.type}
        for name in self.__dataclass_fields__:
            if name in ("id", "type"):
                continue
            value = getattr(self, name)
            if value != getattr(defaults, name):
                out[name] = value
        return out


@dataclass
class Edge:
    source: str
    target: str
    when: str | None = None
    label: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Edge":
        if not isinstance(raw, dict):
            raise GraphError(f"edge must be a mapping, got {type(raw).__name__}")
        source = raw.get("from") or raw.get("source")
        target = raw.get("to") or raw.get("target")
        if not source or not target:
            raise GraphError(f"edge needs 'from' and 'to': {raw!r}")
        unknown = set(raw) - {"from", "to", "source", "target", "when", "label"}
        if unknown:
            raise GraphError(
                f"edge {source}->{target}: unknown key(s) {', '.join(sorted(unknown))}"
            )
        when = raw.get("when")
        return cls(
            source=source,
            target=target,
            when=None if when is None else _as_label(when),
            label=raw.get("label"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"from": self.source, "to": self.target}
        if self.when is not None:
            out["when"] = self.when
        if self.label is not None:
            out["label"] = self.label
        return out

    def display_label(self) -> str | None:
        if self.label:
            return self.label
        return self.when


@dataclass
class Defaults:
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str = DEFAULT_EFFORT
    max_visits: int = DEFAULT_MAX_VISITS
    system: str | None = None
    #: Resident agents keep at most this many exchanges. None keeps everything.
    keep_last: int | None = None
    #: Rendered prompts are trimmed to this many characters. None never trims.
    max_input_chars: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "Defaults":
        raw = raw or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise GraphError(f"defaults: unknown key(s) {', '.join(sorted(unknown))}")
        return cls(**raw)


@dataclass
class Limits:
    max_steps: int = DEFAULT_MAX_STEPS
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_node_visits: int = DEFAULT_MAX_NODE_VISITS
    max_parallel: int = DEFAULT_MAX_PARALLEL

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "Limits":
        raw = raw or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise GraphError(f"limits: unknown key(s) {', '.join(sorted(unknown))}")
        return cls(**raw)


@dataclass
class Graph:
    name: str
    nodes: list[Node]
    edges: list[Edge]
    description: str | None = None
    defaults: Defaults = field(default_factory=Defaults)
    limits: Limits = field(default_factory=Limits)
    vars: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_id = {n.id: n for n in self.nodes}
        #: Where this graph was loaded from, when it came off disk.
        self.path: Path | None = getattr(self, "path", None)

    # -- lookup helpers -------------------------------------------------

    def node(self, node_id: str) -> Node:
        try:
            return self._by_id[node_id]
        except KeyError:
            raise GraphError(f"no such node: {node_id!r}") from None

    def has(self, node_id: str) -> bool:
        return node_id in self._by_id

    def incoming(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target == node_id]

    def outgoing(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_id]

    def inputs(self) -> list[Node]:
        return [n for n in self.nodes if n.type == "input"]

    def outputs(self) -> list[Node]:
        return [n for n in self.nodes if n.type == "output"]

    def max_visits_for(self, node: Node) -> int:
        return node.max_visits if node.max_visits is not None else self.defaults.max_visits

    # -- serialization --------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Graph":
        if not isinstance(raw, dict):
            raise GraphError("graph must be a mapping at the top level")
        unknown = set(raw) - {
            "name",
            "description",
            "defaults",
            "limits",
            "vars",
            "nodes",
            "edges",
        }
        if unknown:
            raise GraphError(f"graph: unknown key(s) {', '.join(sorted(unknown))}")
        nodes_raw = raw.get("nodes")
        if not isinstance(nodes_raw, list) or not nodes_raw:
            raise GraphError("graph needs a non-empty 'nodes' list")
        edges_raw = raw.get("edges") or []
        if not isinstance(edges_raw, list):
            raise GraphError("'edges' must be a list")
        graph = cls(
            name=raw.get("name") or "unnamed-graph",
            description=raw.get("description"),
            defaults=Defaults.from_dict(raw.get("defaults")),
            limits=Limits.from_dict(raw.get("limits")),
            vars=raw.get("vars") or {},
            nodes=[Node.from_dict(n) for n in nodes_raw],
            edges=[Edge.from_dict(e) for e in edges_raw],
        )
        graph.validate()
        return graph

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.description:
            out["description"] = self.description
        defaults = {
            k: v
            for k, v in vars(self.defaults).items()
            if v != getattr(Defaults(), k)
        }
        if defaults:
            out["defaults"] = defaults
        limits = {
            k: v for k, v in vars(self.limits).items() if v != getattr(Limits(), k)
        }
        if limits:
            out["limits"] = limits
        if self.vars:
            out["vars"] = self.vars
        out["nodes"] = [n.to_dict() for n in self.nodes]
        out["edges"] = [e.to_dict() for e in self.edges]
        return out

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, width=100)

    def to_file(self, path: str | Path) -> Path:
        """Write this graph out as YAML (or JSON, by suffix)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        else:
            path.write_text(self.to_yaml())
        return path

    @classmethod
    def from_file(cls, path: str | Path) -> "Graph":
        """Load and validate a graph from a `.yaml`/`.yml`/`.json` file."""
        return load_graph(path)

    @classmethod
    def from_yaml(cls, text: str) -> "Graph":
        """Load and validate a graph from a YAML string."""
        return load_graph_str(text)

    @classmethod
    def from_json(cls, text: str) -> "Graph":
        """Load and validate a graph from a JSON string."""
        return load_graph_str(text, fmt="json")

    def lint(self) -> list["LintWarning"]:
        """Non-fatal warnings — nodes nothing reads, references that never fill.

        See `graph_looper.lint` for the checks.
        """
        from graph_looper.lint import lint as run_lint

        return run_lint(self)

    # -- validation -----------------------------------------------------

    def validate(self) -> None:
        errors: list[str] = []
        seen: set[str] = set()
        for node in self.nodes:
            if node.id in seen:
                errors.append(f"duplicate node id {node.id!r}")
            seen.add(node.id)
            errors.extend(self._validate_node(node))

        for edge in self.edges:
            if edge.source not in seen:
                errors.append(f"edge {edge.source}->{edge.target}: unknown source")
            if edge.target not in seen:
                errors.append(f"edge {edge.source}->{edge.target}: unknown target")

        if not self.inputs():
            errors.append("graph needs at least one node of type 'input'")
        if not self.outputs():
            errors.append("graph needs at least one node of type 'output'")

        errors.extend(self._validate_routing())
        errors.extend(self._validate_reachability())

        if errors:
            raise GraphError(
                "invalid graph:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def _validate_node(self, node: Node) -> list[str]:
        errors: list[str] = []
        where = f"node {node.id!r}"
        if node.type not in NODE_TYPES:
            errors.append(f"{where}: type must be one of {', '.join(NODE_TYPES)}")
        if node.join not in JOIN_MODES:
            errors.append(f"{where}: join must be one of {', '.join(JOIN_MODES)}")
        if node.type == "agent" and node.agent_mode not in AGENT_MODES:
            errors.append(f"{where}: mode must be one of {', '.join(AGENT_MODES)}")
        if node.calls_model() and not node.prompt:
            errors.append(f"{where}: {node.type} nodes need a 'prompt'")
        if node.type == "gate":
            if node.gate_mode not in GATE_MODES:
                errors.append(f"{where}: mode must be one of {', '.join(GATE_MODES)}")
            if len(node.choices) < 2:
                errors.append(f"{where}: gate needs at least two 'choices'")
            if not all(isinstance(c, str) for c in node.choices):
                errors.append(f"{where}: every gate choice must be a string")
            if node.on_exhausted and node.on_exhausted not in node.choices:
                errors.append(
                    f"{where}: on_exhausted {node.on_exhausted!r} is not one of "
                    f"{node.choices}"
                )
            if not node.on_exhausted and node.choices:
                errors.append(
                    f"{where}: gate needs 'on_exhausted' naming the choice to force "
                    "once its visit budget is spent"
                )
            if node.default and node.default not in node.choices:
                errors.append(
                    f"{where}: default {node.default!r} is not one of {node.choices}"
                )
            errors.extend(self._validate_rules(node))
        if node.type != "gate" and (node.rules or node.default):
            errors.append(f"{where}: 'rules' and 'default' only apply to gates")
        if node.state_limit is not None and node.state_limit < 1:
            errors.append(f"{where}: state_limit must be >= 1")
        if node.state_append and not node.writes_state:
            errors.append(f"{where}: state_append needs 'writes_state'")
        if node.writes_state and node.writes_state.startswith("_"):
            errors.append(
                f"{where}: state keys starting with '_' are reserved for the engine"
            )
        if node.keep_last is not None and node.keep_last < 1:
            errors.append(f"{where}: keep_last must be >= 1")
        if node.max_input_chars is not None and node.max_input_chars < 1:
            errors.append(f"{where}: max_input_chars must be >= 1")
        if node.type == "transform" and node.op not in TRANSFORM_OPS:
            errors.append(f"{where}: op must be one of {', '.join(TRANSFORM_OPS)}")
        if node.type == "transform" and node.op == "template" and not node.prompt:
            errors.append(f"{where}: transform op 'template' needs a 'prompt'")
        if node.max_visits is not None and node.max_visits < 1:
            errors.append(f"{where}: max_visits must be >= 1")
        if node.type == "input" and self.incoming(node.id):
            errors.append(f"{where}: input nodes cannot have incoming edges")
        if node.type == "output" and self.outgoing(node.id):
            errors.append(f"{where}: output nodes cannot have outgoing edges")
        if node.type != "input" and not self.incoming(node.id):
            errors.append(f"{where}: no incoming edges — it can never fire")
        if node.type != "output" and not self.outgoing(node.id):
            errors.append(f"{where}: no outgoing edges — its result goes nowhere")
        return errors

    def _validate_rules(self, node: Node) -> list[str]:
        """Predicate rules are compiled here so a bad regex fails at load time,
        not three nodes into a live run."""
        errors: list[str] = []
        where = f"node {node.id!r}"
        if node.gate_mode == "predicate" and not node.rules:
            errors.append(f"{where}: a predicate gate needs 'rules'")
        if node.rules and node.gate_mode != "predicate":
            errors.append(f"{where}: 'rules' needs mode: predicate")
        for index, rule in enumerate(node.rules):
            at = f"{where}: rule {index + 1}"
            if not isinstance(rule, dict):
                errors.append(f"{at} must be a mapping")
                continue
            unknown = set(rule) - set(MATCHERS) - {"choice"}
            if unknown:
                errors.append(f"{at}: unknown key(s) {', '.join(sorted(unknown))}")
            matchers = [m for m in MATCHERS if m in rule]
            if len(matchers) != 1:
                errors.append(
                    f"{at} needs exactly one of {', '.join(MATCHERS)}, "
                    f"got {len(matchers)}"
                )
            choice = rule.get("choice")
            if choice is None:
                errors.append(f"{at} needs a 'choice'")
            elif choice not in node.choices:
                errors.append(f"{at}: choice {choice!r} is not one of {node.choices}")
            if "matches" in rule:
                try:
                    re.compile(str(rule["matches"]))
                except re.error as exc:
                    errors.append(f"{at}: 'matches' is not a valid regex ({exc})")
        return errors

    def _validate_routing(self) -> list[str]:
        errors: list[str] = []
        for node in self.nodes:
            if node.type != "gate":
                continue
            out = self.outgoing(node.id)
            covered = {e.when for e in out if e.when is not None}
            missing = [c for c in node.choices if c not in covered]
            if missing:
                errors.append(
                    f"node {node.id!r}: gate choice(s) {', '.join(missing)} have no "
                    "outgoing edge"
                )
            stray = [
                e.when for e in out if e.when is not None and e.when not in node.choices
            ]
            if stray:
                errors.append(
                    f"node {node.id!r}: edge condition(s) {', '.join(sorted(stray))} "
                    f"are not gate choices {node.choices}"
                )
        return errors

    def _validate_reachability(self) -> list[str]:
        errors: list[str] = []
        reachable = self._reachable_from({n.id for n in self.inputs()})
        orphans = sorted({n.id for n in self.nodes} - reachable)
        if orphans:
            errors.append(
                f"unreachable from any input node: {', '.join(orphans)}"
            )
        terminals = {n.id for n in self.outputs()}
        dead_ends = sorted(
            n.id
            for n in self.nodes
            if n.id in reachable and not self._can_reach(n.id, terminals)
        )
        if dead_ends:
            errors.append(
                f"cannot reach any output node from: {', '.join(dead_ends)}"
            )
        return errors

    def _reachable_from(self, starts: Iterable[str]) -> set[str]:
        seen = set(starts)
        stack = list(seen)
        while stack:
            current = stack.pop()
            for edge in self.outgoing(current):
                if edge.target not in seen:
                    seen.add(edge.target)
                    stack.append(edge.target)
        return seen

    def _can_reach(self, start: str, targets: set[str]) -> bool:
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            if current in targets:
                return True
            for edge in self.outgoing(current):
                if edge.target not in seen:
                    seen.add(edge.target)
                    stack.append(edge.target)
        return False


def load_graph_str(text: str, *, fmt: str = "yaml") -> Graph:
    """Parse a graph from a YAML or JSON string."""
    if fmt == "json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    if raw is None:
        raise GraphError("empty graph document")
    return Graph.from_dict(raw)


def load_graph(path: str | Path) -> Graph:
    """Load and validate a graph from a .yaml/.yml/.json file."""
    path = Path(path)
    if not path.exists():
        raise GraphError(f"no such graph file: {path}")
    fmt = "json" if path.suffix.lower() == ".json" else "yaml"
    try:
        graph = load_graph_str(path.read_text(), fmt=fmt)
    except GraphError as exc:
        raise GraphError(f"{path}: {exc}") from None
    graph.path = path
    return graph
