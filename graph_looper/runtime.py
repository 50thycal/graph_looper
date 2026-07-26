"""The execution engine.

Messages travel along edges. A node fires when its join condition is satisfied
by the messages waiting on its incoming edges, which is what makes cycles work:
the worker in a review loop has two incoming edges (from the planner, and from
the feedback gate) and `join: any`, so it fires on whichever arrives.

Every node that is ready in a given tick runs concurrently, so three reviewers
fan out for the price of the slowest one.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from graph_looper import predicate, template
from graph_looper.providers import (
    AnthropicProvider,
    LLMRequest,
    LLMResponse,
    Provider,
    ProviderError,
    choice_schema,
)
from graph_looper.spec import Graph, Node
from graph_looper.state import StateError, StateStore, record, resolve_store


class RunError(RuntimeError):
    """The run could not be completed."""


@dataclass
class NodeResult:
    node_id: str
    text: str
    visit: int
    data: dict[str, Any] | None = None
    label: str | None = None
    forced: bool = False
    #: How the result was produced: `model`, `predicate`, `forced`, or `local`.
    #: Anything other than `model` cost no tokens.
    via: str = "model"
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = {
            "node": self.node_id,
            "visit": self.visit,
            "text": self.text,
            "via": self.via,
            "seconds": round(self.seconds, 3),
        }
        if self.data is not None:
            out["data"] = self.data
        if self.label is not None:
            out["label"] = self.label
        if self.forced:
            out["forced"] = True
        if self.input_tokens or self.output_tokens:
            out["tokens"] = {
                "input": self.input_tokens,
                "output": self.output_tokens,
            }
        return out


@dataclass
class TraceEvent:
    seq: int
    at: float
    kind: str
    node: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "seq": self.seq,
            "at": round(self.at, 3),
            "kind": self.kind,
        }
        if self.node:
            out["node"] = self.node
        if self.detail:
            out.update(self.detail)
        return out


@dataclass
class Trace:
    events: list[TraceEvent] = field(default_factory=list)

    def add(self, kind: str, node: str | None = None, **detail: Any) -> TraceEvent:
        event = TraceEvent(
            seq=len(self.events), at=time.monotonic(), kind=kind, node=node, detail=detail
        )
        self.events.append(event)
        return event

    def rebase(self) -> None:
        """Make timestamps relative to the first event."""
        if not self.events:
            return
        start = self.events[0].at
        for event in self.events:
            event.at -= start

    def visited_nodes(self) -> list[str]:
        seen: list[str] = []
        for event in self.events:
            if event.kind == "node_end" and event.node and event.node not in seen:
                seen.append(event.node)
        return seen

    def traversed_edges(self) -> list[tuple[str, str]]:
        seen: list[tuple[str, str]] = []
        for event in self.events:
            if event.kind == "edge":
                pair = (event.detail["from"], event.detail["to"])
                if pair not in seen:
                    seen.append(pair)
        return seen

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]


@dataclass
class RunResult:
    graph: str
    task: str
    ok: bool
    final_node: str | None
    output: str
    results: dict[str, NodeResult]
    trace: Trace
    steps: int
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    #: Per-node totals accumulated across every visit, keyed by node id, each
    #: holding `calls`, `model_calls`, `input`, `output`.
    usage_by_node: dict[str, dict[str, int]] = field(default_factory=dict)
    #: State as it stands after the run, including anything nodes wrote.
    state: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def text_of(self, node_id: str, default: str = "") -> str:
        """The latest text a node produced, or *default* if it never ran."""
        result = self.results.get(node_id)
        return result.text if result else default

    def data_of(self, node_id: str) -> dict[str, Any] | None:
        """The latest structured output of a node, if it had a schema."""
        result = self.results.get(node_id)
        return result.data if result else None

    def costliest(self, limit: int = 5) -> list[tuple[str, dict[str, int]]]:
        """Nodes ordered by tokens spent — which one is the pile."""
        ranked = sorted(
            self.usage_by_node.items(),
            key=lambda kv: kv[1]["input"] + kv[1]["output"],
            reverse=True,
        )
        return ranked[:limit]

    def raise_for_status(self) -> "RunResult":
        """Raise `RunError` if the run failed. Handy for calling code that
        would rather have an exception than check a flag."""
        if not self.ok:
            raise RunError(self.error or "run failed")
        return self

    def to_dict(self, *, include_trace: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "graph": self.graph,
            "task": self.task,
            "ok": self.ok,
            "final_node": self.final_node,
            "output": self.output,
            "steps": self.steps,
            "seconds": round(self.seconds, 3),
            "tokens": {"input": self.input_tokens, "output": self.output_tokens},
            "usage_by_node": self.usage_by_node,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }
        if self.state:
            out["state"] = self.state
        if self.error:
            out["error"] = self.error
        if include_trace:
            out["trace"] = self.trace.to_list()
        return out


class Runner:
    """Executes a `Graph` against a `Provider`."""

    def __init__(
        self,
        graph: Graph,
        provider: Provider | None = None,
        *,
        state: Any = None,
        namespace: str | None = None,
        on_event: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        """
        Args:
            graph: the workflow to execute.
            provider: where model calls go. Defaults to `AnthropicProvider()`.
            state: persistence between runs — a `StateStore`, a path, `True` for
                the default file, or `None` for no memory at all.
            namespace: the key state is filed under. Defaults to the graph name,
                so one store can back many graphs.
            on_event: called with each `TraceEvent` as it happens.
        """
        self.graph = graph
        self.provider = provider if provider is not None else AnthropicProvider()
        self.store: StateStore = resolve_store(state)
        self.namespace = namespace or graph.name
        self.on_event = on_event

    def run(self, task: str, **overrides: Any) -> RunResult:
        """Synchronous entry point."""
        return asyncio.run(self.arun(task, **overrides))

    async def arun(self, task: str, *, variables: dict[str, Any] | None = None) -> RunResult:
        graph = self.graph
        trace = Trace()
        self._emit(trace, "run_start", detail={"graph": graph.name, "task": task})

        mailboxes: dict[int, list[NodeResult]] = {i: [] for i in range(len(graph.edges))}
        visits: dict[str, int] = {n.id: 0 for n in graph.nodes}
        results: dict[str, NodeResult] = {}
        usage: dict[str, dict[str, int]] = {}
        conversations: dict[str, list[dict[str, Any]]] = {}
        merged_vars = {**graph.vars, **(variables or {})}
        state = self.store.load(self.namespace)

        pending_inputs = [n for n in graph.inputs()]
        started = time.monotonic()
        steps = 0
        final_node: str | None = None
        output_text = ""
        error: str | None = None
        semaphore = asyncio.Semaphore(max(1, graph.limits.max_parallel))

        try:
            while True:
                elapsed = time.monotonic() - started
                if elapsed > graph.limits.max_seconds:
                    raise RunError(
                        f"time limit reached ({graph.limits.max_seconds}s) after "
                        f"{steps} steps"
                    )
                if steps >= graph.limits.max_steps:
                    raise RunError(
                        f"step limit reached ({graph.limits.max_steps}); the graph is "
                        "probably looping without a way out"
                    )

                if pending_inputs:
                    ready: list[tuple[Node, dict[str, NodeResult]]] = [
                        (node, {}) for node in pending_inputs
                    ]
                    pending_inputs = []
                else:
                    ready = self._collect_ready(mailboxes)

                if not ready:
                    raise RunError(self._stall_report(mailboxes, results))

                steps += len(ready)
                fired = await asyncio.gather(
                    *(
                        self._fire(
                            node,
                            inbox,
                            task=task,
                            variables=merged_vars,
                            visits=visits,
                            results=results,
                            state=state,
                            conversations=conversations,
                            trace=trace,
                            semaphore=semaphore,
                        )
                        for node, inbox in ready
                    )
                )

                terminal: NodeResult | None = None
                for result in fired:
                    results[result.node_id] = result
                    self._tally(usage, result)
                    node = graph.node(result.node_id)
                    if node.writes_state:
                        record(
                            state,
                            node.writes_state,
                            result.data if result.data is not None else result.text,
                            append=node.state_append,
                            limit=node.state_limit,
                        )
                    if node.type == "output":
                        if terminal is None:
                            terminal = result
                        continue
                    self._deliver(node, result, mailboxes, trace)

                if terminal is not None:
                    final_node = terminal.node_id
                    output_text = terminal.text
                    break

        except (RunError, ProviderError) as exc:
            error = str(exc)
            self._emit(trace, "error", detail={"message": error})

        seconds = time.monotonic() - started

        # Persist even on failure: a run that died halfway still learned
        # whatever its completed nodes wrote.
        state["_runs"] = int(state.get("_runs", 0)) + 1
        state["_last_ok"] = error is None
        state["_last_output"] = output_text
        if error:
            state["_last_error"] = error
        try:
            self.store.save(self.namespace, state)
        except StateError as exc:
            self._emit(trace, "warning", detail={"message": f"state not saved: {exc}"})

        self._emit(
            trace,
            "run_end",
            detail={"ok": error is None, "steps": steps, "seconds": round(seconds, 3)},
        )
        trace.rebase()

        return RunResult(
            graph=graph.name,
            task=task,
            ok=error is None,
            final_node=final_node,
            output=output_text,
            results=results,
            trace=trace,
            steps=steps,
            seconds=seconds,
            input_tokens=sum(u["input"] for u in usage.values()),
            output_tokens=sum(u["output"] for u in usage.values()),
            usage_by_node=usage,
            state=state,
            error=error,
        )

    @staticmethod
    def _tally(usage: dict[str, dict[str, int]], result: NodeResult) -> None:
        """Accumulate across every visit — a node in a loop bills more than once."""
        entry = usage.setdefault(
            result.node_id, {"calls": 0, "model_calls": 0, "input": 0, "output": 0}
        )
        entry["calls"] += 1
        if result.via == "model":
            entry["model_calls"] += 1
        entry["input"] += result.input_tokens
        entry["output"] += result.output_tokens

    # -- scheduling ---------------------------------------------------------

    def _collect_ready(
        self, mailboxes: dict[int, list[NodeResult]]
    ) -> list[tuple[Node, dict[str, NodeResult]]]:
        ready: list[tuple[Node, dict[str, NodeResult]]] = []
        for node in self.graph.nodes:
            if node.type == "input":
                continue
            incoming = [
                (i, e) for i, e in enumerate(self.graph.edges) if e.target == node.id
            ]
            filled = [(i, e) for i, e in incoming if mailboxes[i]]
            if not filled:
                continue
            if node.join == "all" and len(filled) != len(incoming):
                continue
            inbox: dict[str, NodeResult] = {}
            for index, edge in filled:
                inbox[edge.source] = mailboxes[index].pop(0)
            ready.append((node, inbox))
        return ready

    def _deliver(
        self,
        node: Node,
        result: NodeResult,
        mailboxes: dict[int, list[NodeResult]],
        trace: Trace,
    ) -> None:
        delivered = 0
        for index, edge in enumerate(self.graph.edges):
            if edge.source != node.id:
                continue
            if edge.when is not None and edge.when != result.label:
                continue
            mailboxes[index].append(result)
            delivered += 1
            self._emit(
                trace,
                "edge",
                detail={
                    "from": edge.source,
                    "to": edge.target,
                    "when": edge.when,
                },
            )
        if delivered == 0:
            self._emit(
                trace,
                "dropped",
                node=node.id,
                detail={
                    "label": result.label,
                    "message": "no outgoing edge matched this result",
                },
            )

    def _stall_report(
        self, mailboxes: dict[int, list[NodeResult]], results: dict[str, NodeResult]
    ) -> str:
        waiting: list[str] = []
        for node in self.graph.nodes:
            if node.type == "input":
                continue
            incoming = [
                (i, e) for i, e in enumerate(self.graph.edges) if e.target == node.id
            ]
            missing = [e.source for i, e in incoming if not mailboxes[i]]
            if missing and len(missing) < len(incoming):
                waiting.append(f"{node.id} still waiting on {', '.join(missing)}")
        detail = "; ".join(waiting) if waiting else "no node has any pending input"
        return (
            "the run stalled before reaching an output node "
            f"(fired: {', '.join(results) or 'nothing'}). {detail}."
        )

    # -- node execution -----------------------------------------------------

    async def _fire(
        self,
        node: Node,
        inbox: dict[str, NodeResult],
        *,
        task: str,
        variables: dict[str, Any],
        visits: dict[str, int],
        results: dict[str, NodeResult],
        state: dict[str, Any],
        conversations: dict[str, list[dict[str, Any]]],
        trace: Trace,
        semaphore: asyncio.Semaphore,
    ) -> NodeResult:
        visits[node.id] += 1
        visit = visits[node.id]
        cap = self._visit_cap(node)
        started = time.monotonic()

        self._emit(
            trace,
            "node_start",
            node=node.id,
            detail={"type": node.type, "visit": visit, "inputs": sorted(inbox)},
        )

        if node.type != "gate" and visit > cap:
            raise RunError(
                f"node {node.id!r} fired {visit} times (cap {cap}); "
                "raise its max_visits or tighten the loop that feeds it"
            )

        context = self._context(node, inbox, task, variables, results, state, visit)

        if node.type == "gate" and visit > cap:
            result = NodeResult(
                node_id=node.id,
                text=(
                    f"Visit budget of {cap} spent; forcing "
                    f"'{node.on_exhausted}' without asking the model."
                ),
                visit=visit,
                data={"choice": node.on_exhausted, "reason": "visit budget exhausted"},
                label=node.on_exhausted,
                forced=True,
                via="forced",
                seconds=time.monotonic() - started,
            )
        elif node.type == "gate" and node.gate_mode == "predicate":
            decision = self._decide(node, context)
            if decision is None:
                # Nothing matched and no default — escalate to the model.
                async with semaphore:
                    response = await self._call_model(node, context, conversations, visit)
                result = NodeResult(
                    node_id=node.id,
                    text=response.text,
                    visit=visit,
                    data=response.data,
                    via="model",
                    seconds=time.monotonic() - started,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
                result.label = self._label_for(node, response.text, response.data)
            else:
                result = NodeResult(
                    node_id=node.id,
                    text=decision.reason,
                    visit=visit,
                    data={"choice": decision.choice, "reason": decision.reason},
                    label=decision.choice,
                    via="predicate",
                    seconds=time.monotonic() - started,
                )
        elif node.type == "input":
            result = NodeResult(
                node_id=node.id,
                text=template.render(node.prompt, context) if node.prompt else task,
                visit=visit,
                via="local",
                seconds=time.monotonic() - started,
            )
        elif node.type == "transform":
            result = NodeResult(
                node_id=node.id,
                text=self._transform(node, inbox, context),
                visit=visit,
                via="local",
                seconds=time.monotonic() - started,
            )
            result.label = self._label_for(node, result.text, None)
        elif node.type == "output":
            text = (
                template.render(node.prompt, context)
                if node.prompt
                else self._joined(inbox)
            )
            result = NodeResult(
                node_id=node.id,
                text=text,
                visit=visit,
                via="local",
                seconds=time.monotonic() - started,
            )
        else:
            async with semaphore:
                response = await self._call_model(node, context, conversations, visit)
            result = NodeResult(
                node_id=node.id,
                text=response.text,
                visit=visit,
                data=response.data,
                via="model",
                seconds=time.monotonic() - started,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            result.label = self._label_for(node, response.text, response.data)

        self._emit(
            trace,
            "node_end",
            node=node.id,
            detail={
                "visit": visit,
                "label": result.label,
                "forced": result.forced,
                "via": result.via,
                "seconds": round(result.seconds, 3),
                "preview": _preview(result.text),
            },
        )
        return result

    def _decide(self, node: Node, context: dict[str, Any]) -> predicate.Decision | None:
        """Run a predicate gate's rules. `None` means escalate to the model."""
        source = node.source if node.source is not None else "{{ input }}"
        text = template.render(source, context)
        decision = predicate.evaluate(node.rules, text)
        if decision.matched:
            return decision
        if node.default:
            return predicate.Decision(
                choice=node.default, reason=f"{decision.reason}; using default"
            )
        return None

    async def _call_model(
        self,
        node: Node,
        context: dict[str, Any],
        conversations: dict[str, list[dict[str, Any]]],
        visit: int,
    ) -> LLMResponse:
        graph = self.graph
        budget = (
            node.max_input_chars
            if node.max_input_chars is not None
            else graph.defaults.max_input_chars
        )
        prompt = template.truncate_middle(
            template.render(node.prompt or "", context), budget
        )
        system = node.system if node.system is not None else graph.defaults.system
        system = template.render(system, context) if system else None

        schema = node.output_schema
        if node.type == "gate":
            schema = schema or choice_schema(node.choices)

        resident = node.type == "agent" and node.agent_mode == "resident"
        history = conversations.setdefault(node.id, []) if resident else []
        if resident:
            keep = node.keep_last if node.keep_last is not None else graph.defaults.keep_last
            if keep is not None and len(history) > keep * 2:
                # One exchange is a user turn plus its assistant reply. Dropping
                # from the front keeps the most recent attempts, which is what a
                # revision loop actually needs.
                del history[: len(history) - keep * 2]
        messages = [*history, {"role": "user", "content": prompt}]

        request = LLMRequest(
            node_id=node.id,
            messages=messages,
            system=system,
            model=node.model or graph.defaults.model,
            max_tokens=node.max_tokens or graph.defaults.max_tokens,
            effort=node.effort or graph.defaults.effort,
            thinking=node.thinking,
            schema=schema,
        )
        response = await self.provider.complete(request)

        if resident:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": response.text})

        return response

    def _transform(
        self, node: Node, inbox: dict[str, NodeResult], context: dict[str, Any]
    ) -> str:
        texts = [inbox[key].text for key in sorted(inbox)]
        if node.op == "concat":
            return node.separator.join(t for t in texts if t)
        if node.op == "first":
            return texts[0] if texts else ""
        if node.op == "last":
            return texts[-1] if texts else ""
        if node.op == "template":
            return template.render(node.prompt or "", context)
        if node.op == "json":
            return json.dumps(
                {k: (v.data if v.data is not None else v.text) for k, v in inbox.items()},
                indent=2,
                ensure_ascii=False,
            )
        raise RunError(f"node {node.id!r}: unknown transform op {node.op!r}")

    def _label_for(
        self, node: Node, text: str, data: dict[str, Any] | None
    ) -> str | None:
        if node.type == "gate":
            choice = (data or {}).get("choice")
            if choice not in node.choices:
                raise RunError(
                    f"gate {node.id!r} chose {choice!r}, which is not one of "
                    f"{node.choices}"
                )
            return str(choice)
        if node.label_from:
            value = template.resolve(node.label_from, {"data": data or {}, "text": text})
            return None if value is None else str(value)
        return None

    def _context(
        self,
        node: Node,
        inbox: dict[str, NodeResult],
        task: str,
        variables: dict[str, Any],
        results: dict[str, NodeResult],
        state: dict[str, Any],
        visit: int,
    ) -> dict[str, Any]:
        return {
            "task": task,
            "node": node.id,
            "iteration": visit,
            "vars": variables,
            "state": state,
            "inputs": {k: v.text for k, v in inbox.items()},
            "data": {k: (v.data or {}) for k, v in inbox.items()},
            "input": self._joined(inbox),
            "results": {k: v.text for k, v in results.items()},
            "results_data": {k: (v.data or {}) for k, v in results.items()},
        }

    def _joined(self, inbox: dict[str, NodeResult]) -> str:
        if len(inbox) == 1:
            return next(iter(inbox.values())).text
        parts = []
        for key in sorted(inbox):
            node = self.graph.node(key)
            parts.append(f"## {node.label}\n\n{inbox[key].text}")
        return "\n\n".join(parts)

    def _visit_cap(self, node: Node) -> int:
        if node.max_visits is not None:
            return node.max_visits
        if node.type == "gate":
            return self.graph.defaults.max_visits
        return self.graph.limits.max_node_visits

    def _emit(
        self,
        trace: Trace,
        kind: str,
        node: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        event = trace.add(kind, node, **(detail or {}))
        if self.on_event:
            self.on_event(event)


def _preview(text: str, limit: int = 160) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
