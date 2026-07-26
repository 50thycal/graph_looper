# graph_looper — public API

This is the contract other projects depend on. Everything listed here is stable;
anything not listed is internal and may change in any release.

Current version: **0.2.0**

## Stability promise

- **Public** — every name exported from the top-level `graph_looper` package
  (i.e. present in `graph_looper.__all__`), and the YAML graph schema.
- **Internal** — everything else, including submodule paths. Import
  `from graph_looper import Runner`, not `from graph_looper.runtime import Runner`.
- Public names are not removed or renamed within a major version. New optional
  parameters may be added; existing ones keep their meaning.
- `tests/test_api.py` asserts the surface exists. A change that breaks a
  downstream import fails there first.

## Install

```bash
pip install graph-looper                      # once published
pip install git+https://github.com/50thycal/graph_looper.git    # from source
```

Add to `pyproject.toml`:

```toml
dependencies = ["graph-looper>=0.2,<1.0"]
```

The package ships `py.typed`, so type checkers see full annotations.

---

## Quick start

```python
from graph_looper import run

result = run("reviewer-loop", "Write a 200-word explainer on tail risk.")
print(result.output)
```

Credentials resolve the way the Anthropic SDK resolves them: `ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN`, or an `ant auth login` profile.

---

## Entry points

### `run(graph, task, **options) -> RunResult`

Execute a graph. The only two required arguments are the graph and the task.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `provider` | `Provider` | `AnthropicProvider()` | Where model calls go. Pass `MockProvider()` for tests. |
| `state` | `StateStore \| str \| Path \| bool` | `None` | Persistence between runs. `None` means no memory and no files written. |
| `namespace` | `str` | graph name | The key state is filed under. |
| `variables` | `dict` | `{}` | Values for `{{ vars.* }}`. |
| `on_event` | `Callable[[TraceEvent], None]` | `None` | Live progress callback. |
| `search_paths` | `Sequence[str \| Path]` | `None` | Extra directories to resolve a graph *name* against. |

### `arun(graph, task, **options) -> RunResult`

Same signature, for callers already inside an event loop. `run()` is
`asyncio.run(arun(...))`.

### `load(source, *, search_paths=None) -> Graph`

Returns a validated `Graph` from whatever you have:

| You pass | It does |
|---|---|
| `Graph` | returns it unchanged |
| `Path`, or a string that is an existing path | loads the file |
| a bare name (`"reviewer-loop"`) | resolves against the search path |
| YAML text (contains a newline and `nodes:`) | parses it |
| `dict` | builds from the graph schema |

Raises `GraphError` if it cannot be found, parsed, or validated. Every other
entry point accepts the same sources.

### `validate(graph) -> Graph`

Load and confirm runnable. Raises `GraphError` with every problem listed.

### `lint(graph) -> list[LintWarning]`

Non-fatal warnings — a node whose output nothing reads, a `{{ inputs.x }}` with
no edge behind it, state written but never read back, an unused var. Empty list
means clean.

### `to_mermaid(graph, *, trace=None, direction="LR", legend=True) -> str`

A Mermaid `flowchart`. Pass `result.trace` to highlight the path a run took and
annotate nodes with visit counts.

### `available(search_paths=None) -> dict[str, Path]`

Every graph findable by name.

---

## `RunResult`

What you get back from `run()`.

| Attribute | Type | Meaning |
|---|---|---|
| `ok` | `bool` | Whether the run reached an output node. |
| `output` | `str` | The terminal node's text. Empty if the run failed. |
| `error` | `str \| None` | Why it failed. |
| `final_node` | `str \| None` | Which output node ended the run. |
| `results` | `dict[str, NodeResult]` | The **latest** result per node. |
| `state` | `dict` | State after the run, including what nodes wrote. |
| `usage_by_node` | `dict[str, dict]` | Per node, across **every** visit: `calls`, `model_calls`, `input`, `output`. |
| `input_tokens` / `output_tokens` / `total_tokens` | `int` | Run totals. |
| `steps` / `seconds` | `int` / `float` | How much work it took. |
| `trace` | `Trace` | Every event, in order. |

Methods: `text_of(node_id, default="")`, `data_of(node_id)`,
`costliest(limit=5)`, `raise_for_status()`, `to_dict(include_trace=True)`.

```python
result = run("reviewer-loop", task)
if not result.ok:
    raise SystemExit(result.error)

print(result.output)
print(result.text_of("synthesize"))       # any node's output
print(result.costliest(3))                # which node cost the most
```

---

## State

Without a store a run starts cold every time. With one, a graph compounds.

```python
from graph_looper import run, FileStateStore

store = FileStateStore("workflows/state.json")
run("draft-critique", "First brief.",  state=store)
run("draft-critique", "Second brief.", state=store)   # reads what run one learned
```

| Store | Use |
|---|---|
| `NullStateStore()` | The default. Nothing persists, nothing is written. |
| `FileStateStore(path)` | A JSON file. Atomic writes. Default `.graphloop/state.json`. |
| `MemoryStateStore()` | This process only — tests, or chaining runs in one script. |

`state=` also accepts a path, or `True` for the default file.

**Writing.** A node declares what it contributes:

```yaml
- id: lesson
  type: agent
  writes_state: lessons     # the key
  state_append: true        # push onto a list instead of replacing
  state_limit: 10           # keep only the most recent 10
  prompt: "In one sentence, what should we avoid next time?"
```

The node's structured `data` is stored if it has an `output_schema`, otherwise
its text.

**Reading.** Any prompt: `{{ state.lessons }}`, `{{ state.profile.tone }}`.

**Reserved keys.** The engine writes `_runs`, `_last_ok`, `_last_output`, and
`_last_error` after every run, including failed ones. Keys beginning with `_`
are rejected in graphs.

**Custom backends.** Implement two methods and pass the object as `state=`:

```python
class RedisStateStore:
    def load(self, namespace: str) -> dict: ...
    def save(self, namespace: str, state: dict) -> None: ...
```

---

## Providers

```python
class Provider(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
```

- `AnthropicProvider(client=None)` — real calls. Streams, adaptive thinking,
  structured outputs for gate decisions. Pass your own `AsyncAnthropic` if you
  need custom configuration.
- `MockProvider(responses=..., fallback=...)` — replays scripted answers keyed
  by node id, consumed one call at a time; the last entry repeats once a node's
  script runs out. Records every call in `.calls` / `.calls_for(node_id)`.

```python
from graph_looper import MockProvider, run

provider = MockProvider({
    "worker": ["first draft", "second draft"],
    "checks_pass": [{"choice": "fail", "reason": "..."}, {"choice": "pass", "reason": "..."}],
})
result = run("reviewer-loop", "task", provider=provider)
assert provider.calls_for("worker")[1].prompt.count("feedback")
```

---

## Finding graphs by name

Resolution order: an actual path → `search_paths=` → `GRAPHLOOPER_PATH` →
graphs bundled with the package.

```bash
export GRAPHLOOPER_PATH=~/workflows:./graphs
```

```python
run("my-workflow", task, search_paths=["./graphs"])
```

---

## Errors

| Exception | Raised when |
|---|---|
| `GraphError` | A graph is missing, unparseable, or invalid. |
| `RunError` | A run stalled, hit a limit, or a gate returned an unknown choice. |
| `ProviderError` | A model call failed, was refused, or returned unusable output. |
| `StateError` | A state store could not be read or written. |

`run()` does **not** raise `RunError` — a failed run comes back with
`ok=False` and `error` set. Call `raise_for_status()` if you would rather have
the exception.

---

## Embedding: a worked example

```python
from graph_looper import FileStateStore, GraphError, run

STORE = FileStateStore(".graphloop/state.json")

def review(document: str, *, tone: str = "direct") -> str:
    """Run our house review workflow and return the finished text."""
    try:
        result = run(
            "reviewer-loop",
            document,
            state=STORE,
            variables={"tone": tone},
            search_paths=["workflows"],
        )
    except GraphError as exc:
        raise RuntimeError(f"workflow is misconfigured: {exc}") from exc

    return result.raise_for_status().output
```

For long runs, stream progress instead of blocking silently:

```python
def on_event(event):
    if event.kind == "node_end":
        print(f"{event.node} done", flush=True)

run("reviewer-loop", task, on_event=on_event)
```

---

## The graph schema

See the README for the full node and edge reference. Summary:

| Node type | Purpose |
|---|---|
| `input` | Entry point; receives the task. |
| `agent` | A model call. `mode: ephemeral` (default) or `resident`. |
| `gate` | A branch. `mode: llm` (default) or `predicate` (local rules, no model call). |
| `transform` | No model call. `op:` `concat` · `first` · `last` · `template` · `json`. |
| `output` | Terminal; its text is the run's result. |

Template paths available in every prompt: `{{ task }}`, `{{ inputs.<node> }}`,
`{{ input }}`, `{{ results.<node> }}`, `{{ data.<node>.<key> }}`,
`{{ state.<key> }}`, `{{ vars.<key> }}`, `{{ iteration }}`.

---

## Changelog

### 0.2.0

Added, all backwards compatible:

- **State** — `state=` on `run()`/`Runner`, `writes_state` on nodes,
  `{{ state.* }}` in prompts, three store implementations plus the `StateStore`
  protocol.
- **Predicate gates** — `mode: predicate` with `rules`, routing at zero tokens,
  falling through to the model when nothing matches and no `default` is set.
- **Context budget** — `keep_last` bounds a resident agent's history,
  `max_input_chars` trims rendered prompts.
- **Lint** — `lint()`, `Graph.lint()`, `graphloop validate --strict`.
- **Library surface** — `run`/`arun`/`load`/`validate`/`lint`/`to_mermaid`/
  `available`, `py.typed`, graph resolution by name via `GRAPHLOOPER_PATH`.
- `RunResult.usage_by_node`, `.costliest()`, `.text_of()`, `.data_of()`,
  `.raise_for_status()`, `.total_tokens`.

Fixed:

- Token totals counted only each node's **last** visit, undercounting every
  looping graph. They now accumulate across all visits.

Behaviour changes worth knowing:

- `Runner(graph)` no longer requires a provider — it defaults to
  `AnthropicProvider()`.
- `graphloop examples` lists everything on the search path, not just bundled
  graphs, and the "graph not found" message names them.
