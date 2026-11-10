"""One question in, one answer out.

`answer()` is the seam between the web layer and the agents: Lead -> parallel Researchers -> Writer,
with every budget coming from config.py. app.py does not know any of this exists.

The Lead has no data tools, only `research` and `write_report`. The Researchers have every data
tool but a clean context holding one brief. The Writer has no tools at all, so the only numbers it
can reach are the findings Python put in front of it. That division is what makes a fabricated
number hard rather than merely discouraged.

Every failure here degrades instead of raising: the run's tool calls are always returned, because a
partial answer with its evidence visible is worth more than an exception string.
"""

import asyncio
from dataclasses import dataclass

import litellm
from agents import MaxTurnsExceeded, Runner, Session

import config
import research
import roster
from schemas import Clarification
from hooks import ToolRecorder
from run_context import RunContext

EMPTY = "The model returned an empty answer. Please try again."

OUT_OF_TURNS = (
    "I reached my {turns}-turn limit for one question, so this answer uses only what I had already "
    "gathered. The tool calls below are what I ran. Try a narrower question."
)

# Prefixed onto every clarifying reply so the next turn can see we already asked. The check is in
# Python, not the prompt: a gate that can loop is worse than no gate, and a model asked "have you
# already asked this?" will sometimes say no.
CLARIFY_MARK = "To point the screen at the right names: "

NOTHING_FOUND = (
    "I could not gather evidence for that question. The research ran but no tool returned a usable "
    "result, so there is nothing I can honestly report. The tool calls below show what was tried."
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
    model_id = config.RESEARCHER_MODELS[researcher_model]
    context = RunContext(recorder=recorder, model_id=model_id)

    verdict = await _clarify(question, session, model_id, context)
    if verdict.reply:
        return {"response": verdict.reply, "tool_calls": recorder.tool_calls, "charts": context.charts}

    # A chart or a single lookup does not need a research desk. One analyst with the tools answers
    # it in a few seconds, where the fan-out would cost two researchers and a writer to say the
    # same thing. The gate decides this; nothing downstream has to know which path it took.
    agent = (roster.analyst.clone(model=roster.model(model_id)) if verdict.simple
             else research.lead_agent(model_id))

    try:
        result = await Runner.run(
            agent,
            question,
            session=session,
            hooks=recorder,
            context=context,
            max_turns=config.MAX_TURNS if verdict.simple else config.LEAD_TURNS,
        )
        # The analyst answers in its own words; the Lead answers through the writer, which stores
        # the audited report on the context rather than returning it.
        response = (str(result.final_output or "").strip() if verdict.simple
                    else context.report.strip())
    except MaxTurnsExceeded:
        # The turn cap is a budget, not a crash. If research happened, write it up anyway rather
        # than throwing away evidence the user already paid for.
        response = context.report.strip() or await _salvage(question, context)
    except litellm.RateLimitError:
        response = RATE_LIMITED

    return {
        "response": response or (NOTHING_FOUND if not context.findings else EMPTY),
        "tool_calls": recorder.tool_calls,
        "charts": context.charts,
    }


@dataclass
class Verdict:
    """What the gate decided: answer now, or which way to route."""

    reply: str = ""  # a finished response to return immediately; empty means carry on
    simple: bool = False  # answer with one analyst rather than the research fan-out


async def _clarify(question: str, session: Session | None, model_id: str, context: RunContext) -> Verdict:
    """Decide whether to answer the question, ask about it, or research it.

    Runs before anything is fetched, so a question with no universe in it does not spend a research
    round discovering that. Any failure routes to the full pipeline: a broken gate must let the
    question through rather than block the desk.
    """
    if await _already_asked(session):
        return Verdict()  # absolute anti-loop: never ask twice in one conversation
    try:
        result = await asyncio.wait_for(
            Runner.run(roster.clarifier, question, session=None, context=context, max_turns=1),
            timeout=config.CLARIFIER_TIMEOUT_S,
        )
    except Exception:
        return Verdict()
    verdict = result.final_output
    if not isinstance(verdict, Clarification):
        return Verdict()
    if verdict.ready or not verdict.question.strip():
        return Verdict(simple=verdict.simple)

    # Out of scope is a statement of fact, not a question, so it gets no marker: it must not
    # consume this conversation's one clarification, and there is nothing for the user to answer.
    if verdict.out_of_scope:
        return Verdict(reply=verdict.question.strip())

    reply = CLARIFY_MARK + verdict.question.strip()
    if session is not None:
        # Written to the session by hand because no Runner.run touched it: the next turn has to see
        # both the vague question and what we asked, or it answers an orphaned reply.
        await session.add_items([
            {"role": "user", "content": question},
            {"role": "assistant", "content": reply},
        ])
    return Verdict(reply=reply)


async def _already_asked(session: Session | None) -> bool:
    """Whether this conversation already contains a clarifying question of ours."""
    if session is None:
        return False
    try:
        items = await session.get_items()
    except Exception:
        return False
    return any(CLARIFY_MARK in str(item.get("content", "")) for item in items)


async def _salvage(question: str, context: RunContext) -> str:
    """Write the report the Lead ran out of turns before requesting.

    A lead that burns its turns has usually already done the research; raising here would discard
    findings that cost real time and real tokens to gather.
    """
    if not context.findings:
        return OUT_OF_TURNS.format(turns=config.LEAD_TURNS)
    try:
        return await research._write(question, context)
    except Exception:
        return OUT_OF_TURNS.format(turns=config.LEAD_TURNS)
