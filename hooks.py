"""Records every tool call a run makes, tagged with the agent that made it.

/chat has to report sub-agent calls with an `agent` field, so the name is taken from the agent
the SDK hands us rather than guessed. Make one recorder per request: it holds that run's calls.

In Phase 3 this becomes a subclass of the course context panel's PanelHooks, which records the
same calls and also streams each step to the page. Its sibling is `run_context.RunContext`: that
one carries what a tool drew, this one records what was called.
"""

import json

from agents import RunHooks


class ToolRecorder(RunHooks):
    """Collects {name, args, result, agent} for each finished tool call, in completion order.

    Completion order is not dispatch order once researchers run in parallel, which is why each
    call carries the agent that made it instead of relying on position.
    """

    def __init__(self) -> None:
        self.tool_calls: list[dict] = []

    async def on_tool_end(self, context, agent, tool, result) -> None:
        # context is a ToolContext, carrying this call's raw arguments. No await below, so appending needs no lock.
        self.tool_calls.append({
            "name": tool.name,
            "args": _arguments(context),
            "result": str(result),
            "agent": agent.name,
        })


def _arguments(context) -> dict:
    """This call's arguments as a dict. A model can emit malformed JSON; the transcript must survive it."""
    raw = getattr(context, "tool_arguments", None)
    try:
        return json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"_raw": str(raw)}
