"""The two per-run collection points the model never sees.

`RunContext` carries what a tool drew and what the researchers found; `ToolRecorder` records what
was called. Both are threaded into every `Runner.run` for a request - the context by `context=`,
the recorder by `hooks=` - so the nested researcher runs report into the same transcript as the
Lead, and share one clock.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agents import RunHooks

import config


@dataclass
class RunContext:
    """Per-request state. One instance per orchestrator.answer() call.

    A tool reaches it through a first parameter annotated `RunContextWrapper[RunContext]`, which the
    SDK keeps out of the schema the model sees. That is what `charts` needs: a tool result is
    replayed on every later turn of the session, so figures must never go to the model.
    """

    charts: list[dict] = field(default_factory=list)  # Plotly specs for the page only

    def draw(self, label: str, figure: dict, agent: str = "") -> None:
        """Keep a figure for the page, unless an identical one was already drawn this run.

        Two researchers handed overlapping briefs screen the same universe and return the same
        table, which reached the reader as the same chart twice. Compared on `data` rather than the
        whole figure, because the two copies differed only in a title built from the arguments -
        same rows, same order, drawn twice. Different data is always a different chart.
        """
        if any(c["figure"].get("data") == figure.get("data") for c in self.charts):
            return
        self.charts.append({"symbol": label, "figure": figure, "agent": agent})

    # What the fan-out needs to start a researcher.
    recorder: Any = None  # the run's ToolRecorder, so nested runs record into the same transcript
    model_id: str = ""  # the LiteLLM id the caller picked, so researchers match the chosen model

    # The evidence store, keyed by the id Python minted.
    findings: dict[str, dict] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)

    # Loop control, counted here so the prompt cannot negotiate it.
    report: str = ""  # the writer's finished answer, set by research.write_report
    rate_limited: int = 0  # upstream 429s this run, from the model or a data feed
    rounds_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def note_rate_limit(self) -> None:
        """Record one upstream refusal, from either the model or a data feed."""
        self.rate_limited += 1

    def throttled(self) -> bool:
        """Whether the run has been refused often enough to stop starting new work.

        A rate limit is not always transient: when a per-minute quota is simply gone, retrying
        spends the budget and returns nothing. Past this many, write up what is already gathered
        instead of dispatching researchers that will be refused too.
        """
        return self.rate_limited >= config.MAX_RATE_LIMITS

    def over_budget(self) -> bool:
        """Whether this run has spent its wall-clock budget and should write up what it has."""
        return time.monotonic() - self.started_at > config.TOTAL_BUDGET_S

    def seconds_left(self) -> float:
        """Wall clock remaining, never negative."""
        return max(0.0, config.TOTAL_BUDGET_S - (time.monotonic() - self.started_at))


class ToolRecorder(RunHooks):
    """Collects {name, args, result, agent, started_at, ended_at} per finished tool call.

    Completion order is not dispatch order once researchers run in parallel, so each call carries
    its own agent and clock rather than relying on position. Those timestamps are the only evidence
    the fan-out overlapped: the Lead's transcript shows one `research` call however many it started.
    """

    def __init__(self) -> None:
        self.tool_calls: list[dict] = []
        self._started: dict[int, float] = {}  # keyed by id(context), unique per in-flight call
        self._origin = time.monotonic()

    def elapsed(self) -> float:
        """Seconds since this run began. Public so research.py stamps its synthetic rows on the same clock."""
        return round(time.monotonic() - self._origin, 3)

    async def on_tool_start(self, context, agent, tool) -> None:
        self._started[id(context)] = self.elapsed()

    async def on_tool_end(self, context, agent, tool, result) -> None:
        # No await below, so appending needs no lock.
        self.tool_calls.append({
            "name": tool.name,
            "args": _arguments(context),
            "result": str(result),
            "agent": agent.name,
            "started_at": self._started.pop(id(context), self.elapsed()),
            "ended_at": self.elapsed(),
        })

    def note(self, name: str, agent: str, args: dict, result: str, started_at: float, ended_at: float) -> None:
        """Record a step the SDK cannot see, such as one researcher inside the fan-out.

        The Lead calls `research` once and the SDK reports one tool call, so without this the
        transcript would claim four researchers ran as a single step.
        """
        self.tool_calls.append({
            "name": name, "args": args, "result": result, "agent": agent,
            "started_at": started_at, "ended_at": ended_at,
        })


def _arguments(context) -> dict:
    """This call's arguments as a dict. A model can emit malformed JSON; the transcript must survive it."""
    raw = getattr(context, "tool_arguments", None)
    try:
        return json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"_raw": str(raw)}
