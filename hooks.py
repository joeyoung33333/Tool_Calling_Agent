"""Records every tool call a run makes, tagged with the agent that made it and when it ran.

/chat has to report sub-agent calls with an `agent` field, so the name is taken from the agent the
SDK hands us rather than guessed. Make one recorder per request: it holds that run's calls.

The timestamps are not decoration. The Lead's own transcript shows a single `research` call no
matter how many researchers it started, so the only way to show the fan-out really overlapped is
per-call `started_at`/`ended_at` that a reader can compare. `research.py` also writes one synthetic
entry per researcher for the same reason.

Its sibling is `run_context.RunContext`: that one carries what a tool drew and what was found,
this one records what was called.
"""

import json
import time

from agents import RunHooks


class ToolRecorder(RunHooks):
    """Collects {name, args, result, agent, started_at, ended_at, seq} per finished tool call.

    Completion order is not dispatch order once researchers run in parallel, which is why each call
    carries both the agent that made it and its own clock, rather than relying on position.
    """

    def __init__(self) -> None:
        self.tool_calls: list[dict] = []
        self._started: dict[int, float] = {}  # keyed by id(context), which is unique per in-flight call
        self._origin = time.monotonic()

    def _now(self) -> float:
        """Seconds since this run began, rounded to milliseconds."""
        return round(time.monotonic() - self._origin, 3)

    async def on_tool_start(self, context, agent, tool) -> None:
        self._started[id(context)] = self._now()

    async def on_tool_end(self, context, agent, tool, result) -> None:
        # context is a ToolContext carrying this call's raw arguments. No await below, so appending needs no lock.
        self.tool_calls.append({
            "name": tool.name,
            "args": _arguments(context),
            "result": str(result),
            "agent": agent.name,
            "started_at": self._started.pop(id(context), self._now()),
            "ended_at": self._now(),
            "seq": len(self.tool_calls),
        })

    def note(self, name: str, agent: str, args: dict, result: str, started_at: float, ended_at: float) -> None:
        """Record a step the SDK cannot see, such as one researcher inside the fan-out.

        The Lead calls `research` once and the SDK reports one tool call, so without this the
        transcript would claim four researchers ran as a single step.
        """
        self.tool_calls.append({
            "name": name, "args": args, "result": result, "agent": agent,
            "started_at": started_at, "ended_at": ended_at, "seq": len(self.tool_calls),
        })


def _arguments(context) -> dict:
    """This call's arguments as a dict. A model can emit malformed JSON; the transcript must survive it."""
    raw = getattr(context, "tool_arguments", None)
    try:
        return json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"_raw": str(raw)}
