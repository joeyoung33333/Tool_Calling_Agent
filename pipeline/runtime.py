"""Per-run state the model never sees: RunContext (charts, findings, budget) and ToolRecorder.

Both are threaded into every Runner.run, so nested researcher runs share one transcript and clock.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agents import RunHooks

import config


@dataclass
class RunContext:
    """Per-request state, one per orchestrator.answer() call, reached through a RunContextWrapper
    that the SDK keeps out of the model's schema, so `charts` never reach the model.
    """

    charts: list[dict] = field(default_factory=list)  # Plotly specs for the page only

    def draw(self, label: str, figure: dict, agent: str = "") -> None:
        """Keep a figure for the page, unless one with identical `data` was drawn this run."""
        if any(c["figure"].get("data") == figure.get("data") for c in self.charts):
            return
        self.charts.append({"symbol": label, "figure": figure, "agent": agent})

    recorder: Any = None  # the run's ToolRecorder, so nested runs record into the same transcript
    model_id: str = ""  # the dropdown's LiteLLM id; only the Researchers read it from here

    # The Clarifier runs without the session, so Python tells it whether this is the first message.
    first_turn: bool = True

    # The evidence store, keyed by the id Python minted.
    findings: dict[str, dict] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)

    # Loop control, counted here so the prompt cannot negotiate it.
    report_question: str = ""  # the Lead's restated question, set by research.write_report
    refused: str = ""  # why research() just declined to run, so its row is marked failed
    rate_limited: int = 0  # upstream 429s this run, from the model or a data feed
    rounds_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def note_rate_limit(self) -> None:
        """Record one upstream refusal, from either the model or a data feed."""
        self.rate_limited += 1

    def throttled(self) -> bool:
        """Whether the run has been refused often enough to stop starting new work."""
        return self.rate_limited >= config.MAX_RATE_LIMITS

    def over_budget(self) -> bool:
        """Whether this run has spent its wall-clock budget and should write up what it has."""
        return time.monotonic() - self.started_at > config.TOTAL_BUDGET_S

    def seconds_left(self) -> float:
        """Wall clock remaining, never negative."""
        return max(0.0, config.TOTAL_BUDGET_S - (time.monotonic() - self.started_at))

    def research_left(self) -> float:
        """Wall clock left for research: the total minus the write-up reserve.

        Researchers are timed against this, not seconds_left(), so a round returns before the cut.
        """
        return max(0.0, self.seconds_left() - config.WRITEUP_RESERVE_S)


class ToolRecorder(RunHooks):
    """Collects {name, args, result, agent, started_at, ended_at} per finished tool call.

    Each call carries its own agent and clock: completion order is not dispatch order in parallel.
    """

    def __init__(self) -> None:
        self.tool_calls: list[dict] = []
        self._started: dict[int, float] = {}  # keyed by id(context), unique per in-flight call
        self._origin = time.monotonic()

    def elapsed(self) -> float:
        """Seconds since this run began; research.py stamps its synthetic rows on this clock."""
        return round(time.monotonic() - self._origin, 3)

    async def on_tool_start(self, context, agent, tool) -> None:
        self._started[id(context)] = self.elapsed()

    async def on_tool_end(self, context, agent, tool, result) -> None:
        # No await below, so appending needs no lock.
        row = {
            "name": tool.name,
            "args": _arguments(context),
            "result": str(result),
            "agent": agent.name,
            "started_at": self._started.pop(id(context), self.elapsed()),
            "ended_at": self.elapsed(),
        }
        error = _error_of(str(result))
        run = getattr(context, "context", None)
        if tool.name == "research" and getattr(run, "refused", ""):
            error, run.refused = run.refused, ""
        if error:
            row["error"] = error  # drawn in red, like a failed agent
        self.tool_calls.append(row)

    def note(self, name: str, agent: str, args: dict, result: str, started_at: float, ended_at: float,
             error: str = "") -> None:
        """Record a step the SDK cannot see, such as one researcher inside the fan-out.

        `error` marks a failed attempt, which the page draws in red; it is omitted when empty.
        """
        row = {"name": name, "args": args, "result": result, "agent": agent,
               "started_at": started_at, "ended_at": ended_at}
        if error:
            row["error"] = error
        self.tool_calls.append(row)


def _error_of(result: str) -> str:
    """The `error` a tool returned as JSON, shortened for a label; empty if it succeeded."""
    if not result.startswith('{"error"'):
        return ""
    try:
        return str(json.loads(result).get("error", ""))[:80]
    except (TypeError, ValueError):
        return ""


def _arguments(context) -> dict:
    """This call's arguments as a dict; malformed JSON from a model must not break this."""
    raw = getattr(context, "tool_arguments", None)
    try:
        return json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"_raw": str(raw)}
