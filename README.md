# graph_looper

Draw an agent workflow as a graph. Run it as a real multi-agent loop.

You write the graph once, as YAML — nodes are agents, gates, and transforms;
edges are the routing, including the edges that point *backwards*. The engine
runs it: fanning out to parallel reviewers, waiting on joins, routing on gate
decisions, looping back on failure, and stopping when a visit budget runs out.
Then it hands you the trace and a diagram of the path it actually took.

It is built to be depended on. One import and one call from any project:

```python
from graph_looper import run

result = run("reviewer-loop", "Write a 200-word explainer on tail risk.")
print(result.output)
```

The full public interface and its stability promise are in
**[docs/API.md](docs/API.md)**.

```mermaid
flowchart LR
  task(["Task"])
  planner["Planner"]
  plan_reviewer["Independent plan reviewer"]
  plan_approved{"Plan approved?"}
  worker["Worker"]
  reviewer_structure["Reviewer 1 — structure"]
  reviewer_correctness["Reviewer 2 — correctness"]
  reviewer_fidelity["Reviewer 3 — task fidelity"]
  synthesize["Synthesise feedback"]
  checks_pass{"All checks pass?"}
  haiku["Turn into haiku"]
  send(("Send to user"))
  task --> planner
  planner --> worker
  planner --> plan_reviewer
  plan_reviewer --> plan_approved
  plan_approved -- "revise plan" --> planner
  plan_approved -- "approve" --> send
  worker -- "1" --> reviewer_structure
  worker -- "2" --> reviewer_correctness
  worker -- "N" --> reviewer_fidelity
  reviewer_structure --> synthesize
  reviewer_correctness --> synthesize
  reviewer_fidelity --> synthesize
  synthesize --> checks_pass
  checks_pass -- "return feedback" --> worker
  checks_pass -- "pass" --> haiku
  haiku --> send

  classDef io fill:#1f2933,stroke:#8899a6,color:#f5f7fa;
  classDef ephemeral fill:#ffffff,stroke:#5b6b7b,color:#1f2933;
  classDef resident fill:#e6f5ea,stroke:#2f8f4e,color:#12351f,stroke-width:2px;
  classDef gate fill:#3b2f10,stroke:#b8860b,color:#f7efd8;
  class task io;
  class planner ephemeral;
  class plan_reviewer ephemeral;
  class plan_approved gate;
  class worker resident;
  class reviewer_structure ephemeral;
  class reviewer_correctness ephemeral;
  class reviewer_fidelity ephemeral;
  class synthesize ephemeral;
  class checks_pass gate;
  class haiku ephemeral;
  class send io;
```

That graph ships as `reviewer-loop`. Green is a **resident** agent — it keeps its
own conversation across visits, so on a retry the worker remembers the draft it
wrote last time. White is **ephemeral** — fresh context every visit, which is
what you want for reviewers who should not be anchored by their own earlier
verdict.

## Install

```bash
pip install -e .                                                 # from a clone
pip install git+https://github.com/50thycal/graph_looper.git     # from anywhere
```

Credentials resolve the way the Anthropic SDK resolves them: `ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN`, or a profile from `ant auth login`.

## Use it from another project

`load()` takes whatever form your graph is already in — a name, a path, YAML
text, a dict, or a `Graph` — so there is nothing to convert:

```python
from graph_looper import FileStateStore, run

result = run(
    "review",                        # a name on your GRAPHLOOPER_PATH…
    document,                        # …or a path, YAML text, or a dict
    state=FileStateStore("state.json"),
    variables={"tone": "direct"},
    search_paths=["workflows"],
)

print(result.output)                 # the terminal node's text
print(result.text_of("synthesize"))  # any node's output
print(result.costliest(3))           # which node spent the most
result.raise_for_status()            # or check result.ok yourself
```

Point it at your own graphs with `GRAPHLOOPER_PATH` (same shape as `PATH`) or
`search_paths=`. `arun()` is the async form. The package ships `py.typed`, so
your type checker sees everything.

## Use it

```bash
graphloop examples                      # what's bundled
graphloop validate reviewer-loop -v     # check a graph, list its loops
graphloop viz reviewer-loop             # Mermaid, to stdout

graphloop run reviewer-loop \
  --input "Write a 200-word explainer on why prediction markets misprice tails." \
  --trace run.json \
  --viz run.md
```

`--trace` writes every event, every node result, and the token count.
`--viz` writes the same diagram with the path taken highlighted and the visit
counts on the nodes — so a worker that went round twice reads `Worker ×2`.

### Run it without spending anything

`--dry-run` swaps in a mock provider, so routing, joins, loops, and budgets all
execute with no model calls. `--mock` scripts specific answers per node, which
is how you test that a loop actually loops:

```bash
cat > mock.json <<'JSON'
{
  "worker": ["draft one", "draft two"],
  "checks_pass": [{"choice": "fail", "reason": "two real defects"},
                  {"choice": "pass", "reason": "all reviewers passed"}]
}
JSON

graphloop run reviewer-loop -i "..." --mock mock.json
```

Each node's script is consumed one call at a time; once it runs out the last
entry repeats, so a loop whose final scripted verdict is `pass` always
terminates.

## Writing a graph

```yaml
name: draft-critique

defaults:
  model: claude-opus-5
  effort: medium
  max_visits: 3          # how many times a gate is asked before it is forced

nodes:
  - id: brief
    type: input

  - id: writer
    type: agent
    mode: resident       # keeps its conversation across visits
    join: any            # fires on whichever edge delivers first
    prompt: |
      Brief: {{ task }}

      Critique of your last draft (blank on the first pass):
      {{ results.critic }}

      Write the next draft. Output the draft only.

  - id: critic
    type: agent          # ephemeral by default — fresh context every visit
    prompt: |
      Draft:
      {{ inputs.writer }}

      Name the three things most worth fixing. If it meets the brief, say
      "Ship it." and nothing else.

  - id: good_enough
    type: gate
    choices: [ship, revise]
    on_exhausted: ship   # forced once the visit budget is spent
    prompt: "{{ inputs.critic }}"

  - id: final
    type: output
    prompt: "{{ results.writer }}"

edges:
  - from: brief
    to: writer
  - from: writer
    to: critic
  - from: critic
    to: good_enough
  - from: good_enough
    to: writer           # the loop
    when: revise
  - from: good_enough
    to: final
    when: ship
```

### Node types

| type        | what it does |
|-------------|--------------|
| `input`     | Entry point. Receives the run's task. |
| `agent`     | An LLM call. `mode: ephemeral` (default) or `resident`. |
| `gate`      | A branch. `mode: llm` (default) asks the model; `mode: predicate` decides locally. |
| `transform` | No model call. `op:` `concat` · `first` · `last` · `template` · `json`. |
| `output`    | Terminal. Its text is the run's result. |

Agent and gate nodes also take `model`, `max_tokens`, `effort`, `system`,
`thinking: false`, and `output_schema` (a JSON schema — the node then returns
parsed structured output, and `label_from: data.<key>` lets a plain agent route
like a gate).

### Gates that cost nothing

Most branches are not judgement calls. A predicate gate decides from local rules
at zero tokens, and only falls through to the model when nothing matches *and*
no `default` is set:

```yaml
- id: classify
  type: gate
  mode: predicate
  choices: [bug, billing, question]
  on_exhausted: question
  source: "{{ task }}"          # what to match against; defaults to {{ input }}
  rules:
    - contains: [refund, invoice, charge]
      choice: billing
    - matches: '\b(error|crash|broken|500)\b'
      choice: bug
  prompt: |                     # reached only when no rule matched
    Keyword routing did not settle this. Classify it: ...
```

Matchers are `contains` (any of a list, case-insensitive), `matches` (regex),
`equals`, `not_empty`, and `always`. First match wins. Regexes are compiled at
load time, so a bad one fails before the run starts rather than three nodes in.

Set `default:` instead of a `prompt` and the gate never reaches the model at all.

### Keeping a lid on context

A resident agent otherwise carries every draft it has ever written into the next
call, which is how a long loop quietly turns into a very expensive one.

```yaml
defaults:
  keep_last: 2            # resident agents keep the last 2 exchanges
  max_input_chars: 20000  # rendered prompts are trimmed from the middle
```

Both can be set per node. `graphloop run --cost` prints tokens by node across
every visit, so you can see which node is the pile.

### Joins — the part that makes loops work

A node fires when its incoming edges satisfy its join mode:

- **`join: all`** (default) — waits for a message on *every* incoming edge.
  This is fan-in: the synthesiser waits for all three reviewers.
- **`join: any`** — fires on whichever arrives first. Any node sitting inside a
  loop needs this, because it is fed both by its normal upstream *and* by the
  feedback edge, and only one of those will deliver on a given pass. `all`
  would deadlock there.

Every node ready in the same tick runs concurrently, so three reviewers cost
the wall-clock of the slowest one, not the sum.

### Loops always terminate

Three independent brakes, so a graph cannot spin forever:

- **Gate visit budget** — `max_visits` on a gate (default from
  `defaults.max_visits`). Once spent, the gate stops asking the model and takes
  its `on_exhausted` branch. This is the one that actually shapes behaviour:
  *retry twice, then ship what we have.*
- **`limits.max_node_visits`** (default 25) — a runaway guard on every other
  node. Tripping it fails the run rather than forcing a branch, because it means
  the graph is wrong, not the work.
- **`limits.max_steps`** / **`limits.max_seconds`** — whole-run ceilings.

A run that cannot reach an output stops with a diagnostic naming the node that
stalled and what it was waiting for, instead of hanging.

### Prompt templates

Prompts are templates over `{{ dotted.paths }}`. Single braces are left alone,
so JSON examples in a prompt survive untouched.

| path | what it is |
|------|-----------|
| `{{ task }}` | the run's input |
| `{{ inputs.<node> }}` | text from that upstream node, *for this firing* |
| `{{ input }}` | every incoming message, joined under headings |
| `{{ results.<node> }}` | that node's most recent result, from anywhere in the run |
| `{{ data.<node>.<key> }}` | a structured field from an upstream node |
| `{{ iteration }}` | this node's visit count |
| `{{ vars.<key> }}` | a variable from `--var key=value` |
| `{{ state.<key> }}` | something a previous **run** recorded |

Unknown paths render empty. That is deliberate: on the first pass through a
loop, `{{ results.synthesize }}` has nothing in it yet, and that is not an error.

Use `inputs.` when a node should react to what just arrived, and `results.` when
it needs the latest of something that arrived on a different pass — which is why
the worker reads `{{ results.planner }}` for the plan and `{{ results.synthesize }}`
for the feedback.

## State: what survives between runs

Without a store, every run starts cold — if the same reviewer raises the same
complaint on ten consecutive runs, nothing anywhere learns it. A store lets a
graph compound.

```yaml
- id: lesson
  type: agent
  writes_state: lessons     # the key it contributes
  state_append: true        # push onto a list rather than replacing
  state_limit: 10           # keep the most recent 10 — a note, not a transcript
  prompt: "In one sentence, what should we avoid next time?"

- id: writer
  type: agent
  prompt: |
    Recurring problems worth avoiding up front:
    {{ state.lessons }}
    ...
```

```bash
graphloop run draft-critique -i "First brief."  --state
graphloop run draft-critique -i "Second brief." --state   # reads run one's lesson
```

The bare `--state` flag uses `.graphloop/state.json`; give it a path to put it
elsewhere. From code, pass `state=` a path, a `FileStateStore`, a
`MemoryStateStore`, or your own object with `load`/`save`.

**Nothing is written unless you ask.** The default is no persistence, so
importing this library never leaves files in someone's working directory. The
engine also records `_runs`, `_last_ok`, `_last_output` and `_last_error` on
every run, including failed ones — a run that dies halfway still keeps whatever
its finished nodes wrote.

## Lint: nodes that do not earn their place

`validate` proves a graph *can* run. `lint` asks whether every node is pulling
its weight — it runs automatically as part of `validate`:

```bash
graphloop validate my-graph            # warnings on stderr
graphloop validate my-graph --strict   # exit 1 if there are any
```

It catches a node whose output no prompt ever reads (it runs, it bills you, the
result is dropped), a `{{ inputs.x }}` with no edge behind it that will always
render blank, state written but never read back, and unused vars. The
`author` command lints what it generates, which is where these mistakes are
most likely.

## Generating a graph from a sketch

```bash
graphloop author "a research loop: three searchers in parallel, a synthesiser, \
  and a gate that sends it back for gaps" -o research.yaml

graphloop author --image whiteboard.jpg -o from-sketch.yaml
```

Claude reads the description or the photo, emits a graph, and the result is
validated. If validation fails, the errors are handed back for another attempt —
so what lands on disk is a graph that runs, not a plausible-looking one.

## Driving the engine directly

`run()` covers most cases; reach for `Runner` when you want to hold the object.

```python
from graph_looper import Graph, MemoryStateStore, Runner

graph = Graph.from_file("workflows/review.yaml")
runner = Runner(graph, state=MemoryStateStore())   # provider defaults to Anthropic

result = runner.run("Write the explainer.")
print(result.output, result.usage_by_node)
```

Full reference: **[docs/API.md](docs/API.md)**.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The suite runs entirely on the mock provider — no key, no network. The live
provider is covered by tests that assert the exact request shape (thinking mode,
effort, structured-output schema, refusal handling) against a fake client.

## Layout

```
graph_looper/
  api.py        the public entry points — run, load, lint, to_mermaid
  spec.py       graph parsing and validation
  runtime.py    the execution engine — scheduling, joins, loops, budgets
  state.py      persistence between runs
  predicate.py  rule matching for gates that skip the model
  lint.py       warnings about nodes that do not earn their place
  catalog.py    finding graphs by name
  providers.py  Anthropic and mock providers
  template.py   the {{ path }} renderer
  render.py     Mermaid and text output
  author.py     graph generation from a description or an image
  cli.py        the graphloop command
  graphs/       bundled example graphs
docs/API.md     the public interface and its stability promise
```
