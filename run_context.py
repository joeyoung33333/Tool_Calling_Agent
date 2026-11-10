"""State shared by everything inside one run, which the model never sees.

`Runner.run(context=...)` threads this through, and a tool reaches it by taking a first parameter
annotated `RunContextWrapper[RunContext]` - which the SDK excludes from the schema the model sees.

This exists because some tool output is for the page, not the model. A tool's return value goes
into the session, so anything bulky returned to the model is replayed on every later turn of the
conversation. `tools._run()` is the only place that writes `charts`.

It also carries what the research fan-out needs: the recorder and model to hand each researcher,
the evidence store the writer is given, and the counters that cap the loop. Those live here rather
than in the prompt because a model cannot be trusted to count its own rounds.

Its sibling is `hooks.ToolRecorder`, the other per-run collection point the model never sees: that
one records what was called, this one carries what was drawn and what was found.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import config


@dataclass
class RunContext:
    """Per-request state. One instance per orchestrator.answer() call."""

    # Plotly specs for the page only: a 250-point line is ~1,700 tokens, re-sent on every later turn if the model saw it.
    charts: list[dict] = field(default_factory=list)

    # --- what the fan-out needs to start a researcher ---
    recorder: Any = None  # the run's ToolRecorder, so nested runs record into the same transcript
    model_id: str = ""  # the LiteLLM id the caller picked, so researchers match the chosen model

    # --- the evidence store, keyed by the id Python minted ---
    findings: dict[str, dict] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)

    # --- loop control, counted in Python so the prompt cannot negotiate it ---
    report: str = ""  # the writer's finished answer, set by research.write_report
    rounds_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def over_budget(self) -> bool:
        """Whether this run has used its wall-clock budget and should write up what it has."""
        return time.monotonic() - self.started_at > config.TOTAL_BUDGET_S

    def seconds_left(self) -> float:
        """Wall clock remaining, never negative."""
        return max(0.0, config.TOTAL_BUDGET_S - (time.monotonic() - self.started_at))
