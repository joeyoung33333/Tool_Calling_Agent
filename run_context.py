"""State shared by everything inside one run, which the model never sees.

`Runner.run(context=...)` threads this through, and a tool reaches it by taking a first parameter
annotated `RunContextWrapper[RunContext]` — which the SDK excludes from the schema the model sees.

This exists because some tool output is for the page, not the model. A tool's return value goes
into the session, so anything bulky returned to the model is replayed on every later turn of the
conversation. `tools._run()` is the only place that writes here. Phase 3 adds the evidence store
and the fan-out counters.

Its sibling is `hooks.ToolRecorder`, the other per-run collection point the model never sees: that
one records what was called, this one carries what was drawn.
"""

from dataclasses import dataclass, field


@dataclass
class RunContext:
    """Per-request state. One instance per orchestrator.answer() call."""

    # Plotly specs for the page only: a 250-point line is ~1,700 tokens, re-sent on every later turn if the model saw it.
    charts: list[dict] = field(default_factory=list)
