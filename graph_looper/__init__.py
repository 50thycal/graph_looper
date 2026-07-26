"""graph_looper — draw an agent workflow as a graph; run it as a real loop.

The one-line version::

    from graph_looper import run

    result = run("reviewer-loop", "Write the Q3 explainer.")
    print(result.output)

With more control::

    from graph_looper import Graph, MockProvider, Runner, FileStateStore

    graph = Graph.from_file("workflows/review.yaml")
    runner = Runner(graph, MockProvider(), state=FileStateStore("state.json"))
    result = runner.run("Write the Q3 explainer.")

Everything exported here is public and covered by the stability promise in
`docs/API.md`. Anything reached through a module path that is not re-exported
below is internal and may change without notice.
"""

from graph_looper.api import (
    arun,
    available,
    lint,
    load,
    run,
    to_mermaid,
    validate,
)
from graph_looper.lint import LintWarning
from graph_looper.predicate import Decision
from graph_looper.providers import (
    AnthropicProvider,
    LLMRequest,
    LLMResponse,
    MockProvider,
    Provider,
    ProviderError,
)
from graph_looper.runtime import (
    NodeResult,
    RunError,
    RunResult,
    Runner,
    Trace,
    TraceEvent,
)
from graph_looper.spec import (
    Defaults,
    Edge,
    Graph,
    GraphError,
    Limits,
    Node,
    load_graph,
    load_graph_str,
)
from graph_looper.state import (
    FileStateStore,
    MemoryStateStore,
    NullStateStore,
    StateError,
    StateStore,
)

__version__ = "0.2.0"

__all__ = [
    # Entry points — start here.
    "run",
    "arun",
    "load",
    "validate",
    "lint",
    "to_mermaid",
    "available",
    # Graph definition.
    "Graph",
    "Node",
    "Edge",
    "Defaults",
    "Limits",
    "load_graph",
    "load_graph_str",
    # Execution.
    "Runner",
    "RunResult",
    "NodeResult",
    "Trace",
    "TraceEvent",
    # Providers.
    "Provider",
    "AnthropicProvider",
    "MockProvider",
    "LLMRequest",
    "LLMResponse",
    # State.
    "StateStore",
    "FileStateStore",
    "MemoryStateStore",
    "NullStateStore",
    # Supporting types.
    "LintWarning",
    "Decision",
    # Errors.
    "GraphError",
    "RunError",
    "ProviderError",
    "StateError",
    "__version__",
]
