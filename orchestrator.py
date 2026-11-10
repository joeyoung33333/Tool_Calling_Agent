"""One question in, one answer out.

`answer()` is the seam between the web layer and the agents. Phase 1 runs a single analyst; Phase 3
replaces the body with the Clarifier -> Lead -> Writer pipeline and the fan-out, and app.py does not
change. Everything that constrains a run (models, turn caps) comes from config.py.

Every failure here degrades instead of raising: the run's tool calls are always returned, because a
partial answer with its evidence visible is worth more than an exception string.
"""

import litellm
from agents import MaxTurnsExceeded, Runner, Session

import roster
import config
from hooks import ToolRecorder
from run_context import RunContext

EMPTY = "The model returned an empty answer. Please try again."

OUT_OF_TURNS = (
    "I reached my {turns}-turn limit for one question, so this answer uses only what I had already "
    "gathered. The tool calls below are what I ran. Try a narrower question."
)

RATE_LIMITED = (
    "Vertex AI is rate limiting this model (HTTP 429: its per-minute quota is used up). Wait a "
    "minute and try again, or pick a different model."
)


async def answer(
    question: str, session: Session | None, researcher_model: str, recorder: ToolRecorder | None = None
) -> dict:
    """Answer one question, continuing `session`'s conversation if there is one.

    Returns the body of /chat minus the session id: the answer, every tool call with the agent that
    made it, and any charts built for the page.

    `recorder` lets the caller supply its own hooks. Phase 3 passes a PanelHooks subclass so the
    live panel sees each step as it happens; the same object must also be handed to the nested
    Runner.run inside the research fan-out, or sub-agent calls would be missing from tool_calls.
    """
    recorder = recorder or ToolRecorder()
    context = RunContext()
    # Cloned per run for the caller's model choice, without mutating the shared agent.
    agent = roster.analyst.clone(model=roster.model(config.RESEARCHER_MODELS[researcher_model]))

    try:
        result = await Runner.run(
            agent,
            question,
            session=session,
            hooks=recorder,
            context=context,
            max_turns=config.MAX_TURNS,
        )
        response = str(result.final_output or "").strip() or EMPTY
    except MaxTurnsExceeded:
        # The turn cap is a budget, not a crash: answer from what was gathered and keep the tool calls already recorded.
        response = OUT_OF_TURNS.format(turns=config.MAX_TURNS)
    except litellm.RateLimitError:
        response = RATE_LIMITED

    return {"response": response, "tool_calls": recorder.tool_calls, "charts": context.charts}
