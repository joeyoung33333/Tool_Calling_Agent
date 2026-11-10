"""One question in, one answer out.

`answer()` is the seam between the web layer and the agents: Lead -> parallel Researchers ->
Writer, with every budget from config.py. app.py does not know any of this exists.

The division of tools is what makes a fabricated number hard rather than merely discouraged: the
Lead holds only `research` and `write_report`, the Researchers hold every data tool but a clean
context with one brief, and the Writer holds nothing, so the only numbers it can reach are the
findings Python put in front of it.

Every failure here degrades rather than raising, and the run's tool calls are always returned: a
partial answer with its evidence visible beats an exception string.
"""

import asyncio
from dataclasses import dataclass

import litellm
from agents import MaxTurnsExceeded, ModelBehaviorError, Runner, Session

import config
from pipeline import citations, research, roster
from pipeline.runtime import RunContext, ToolRecorder
from pipeline.schemas import Clarification

EMPTY = "The model returned an empty answer. Please try again."

# No turn count: _salvage is reached from both the analyst path (MAX_TURNS) and the Lead's
# (LEAD_TURNS), so naming one of them would report the wrong number on the other.
OUT_OF_TURNS = (
    "I hit my turn limit on this question and could not assemble an answer I can stand behind. "
    "The tool calls below show what was tried. Try a narrower question."
)

# Prefixed onto every clarifying reply so the next turn can see we already asked. Checked in
# Python, not the prompt: a model asked "have you already asked this?" will sometimes say no, and a
# gate that can loop is worse than no gate.
CLARIFY_MARK = "To point the screen at the right names: "

NOTHING_FOUND = (
    "I could not produce an answer I can stand behind: no tool returned a usable result, so there "
    "is nothing I can honestly report. The tool calls below show what was tried."
)

OVER_TIME = (
    "_This question hit the {secs}s limit, so the answer below uses only what had been gathered by "
    "then. A narrower question will finish._"
)

RATE_LIMITED = (
    "Vertex AI is rate limiting this model (HTTP 429: its per-minute quota is used up). Wait a "
    "minute and try again, or pick a different model."
)


async def answer(question: str, session: Session | None, researcher_model: str) -> dict:
    """Answer one question, continuing `session`'s conversation if there is one.

    Returns the body of /chat minus the session id: the answer, every tool call with the agent that
    made it, and any charts built for the page.
    """
    recorder = ToolRecorder()
    model_id = config.RESEARCHER_MODELS[researcher_model]
    context = RunContext(recorder=recorder, model_id=model_id)

    # No model_id: the gate is pinned to config.REASONING_MODEL and does not follow the dropdown.
    verdict = await _clarify(question, session, context)
    if verdict.reply:
        return {"response": verdict.reply, "tool_calls": recorder.tool_calls, "charts": context.charts}

    # A chart or a single lookup does not need a research desk; one analyst answers it in seconds.
    # The gate decides, and nothing downstream has to know which path it took.
    agent = (roster.analyst.clone(model=roster.model(model_id)) if verdict.simple
             else research.lead_agent(model_id))

    # The only Runner.run in the project without a wall clock, and the outermost one: every nested
    # call was capped while this was free to run as long as its retries allowed. TOTAL_BUDGET_S was
    # checked inside the research tool, which cannot interrupt a model call already in flight.
    # Reserve a slice so a run that spends everything researching can still write up what it has.
    budget = max(15.0, config.TOTAL_BUDGET_S - config.WRITEUP_RESERVE_S)
    # The top-level agent's own span. Its group was otherwise derived from its tool calls, so the
    # timeline showed a gap between the Clarifier returning and the Lead's first call - the gap was
    # the Lead's first model round-trip, which nothing recorded, and it read as idle time.
    top_started = recorder.elapsed()
    top_name = "Analyst" if verdict.simple else "Lead"
    try:
        result = await asyncio.wait_for(
            Runner.run(
                agent,
                question,
                session=session,
                hooks=recorder,
                context=context,
                max_turns=config.MAX_TURNS if verdict.simple else config.LEAD_TURNS,
            ),
            timeout=budget,
        )
        # The analyst answers in its own words; the Lead answers through the writer, which leaves
        # the audited report on the context rather than returning it.
        if verdict.simple:
            response = _attest(str(result.final_output or "").strip(), recorder.tool_calls, question)
        else:
            response = context.report.strip()
    except MaxTurnsExceeded:
        # The turn cap is a budget, not a crash: write up whatever research already happened.
        response = context.report.strip() or await _salvage(question, context)
    except ModelBehaviorError:
        # The Lead called a data tool it does not hold. Both routes share one session, so a turn
        # the analyst answered leaves its get_price_history call in the history, and the Lead
        # imitates it on a later turn. The analyst does hold that tool, so re-run there rather
        # than showing the user a raw SDK error: the question is answerable, by the other agent.
        response = await _retry_as_analyst(question, session, model_id, recorder, context)
    except asyncio.TimeoutError:
        # Out of wall clock. The findings are still on the context, so write them up rather than
        # returning nothing after the user has already waited the whole budget.
        response = context.report.strip() or await _salvage(question, context)
        if response:
            response = OVER_TIME.format(secs=int(budget)) + "\n\n" + response
    except litellm.RateLimitError:
        response = RATE_LIMITED

    recorder.note(
        name="agent", agent=top_name,
        args={"question": question, "route": "fast path" if verdict.simple else "research desk"},
        result=f"{len(recorder.tool_calls)} tool calls, {len(context.findings)} findings",
        started_at=top_started, ended_at=recorder.elapsed(),
    )
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


def _attest(answer: str, tool_calls: list[dict], question: str = "") -> str:
    """Append the fast path's provenance, and flag any number not present in a tool result.

    With no findings store to cite, the analyst path is held to the check that is available: every
    number must appear in data this run fetched. Unverifiable figures are named, not removed.
    """
    if not answer or not tool_calls:
        return answer
    checked = citations.verify_against_tools(answer, tool_calls, question)
    out = [answer]
    if checked["missing"]:
        out.append(
            "\n_Not found in any tool result from this run, so treat as unverified: "
            + ", ".join(checked["missing"][:8]) + "._"
        )
    out.append(citations.sources_block(tool_calls))
    return "\n".join(part for part in out if part)


async def _clarify(question: str, session: Session | None, context: RunContext) -> Verdict:
    """Decide whether to answer the question, ask about it, or research it.

    Runs before anything is fetched, so a question with no universe does not spend a research round
    discovering that. Any failure routes to the full pipeline rather than blocking the desk.
    """
    if await _already_asked(session):
        return Verdict()  # absolute anti-loop: never ask twice in one conversation
    # The gate has no tools, so the SDK's tool hooks can never see it. Recorded by hand, or the
    # transcript would show every run starting at the Lead and never mention the step that routed it.
    started = context.recorder.elapsed() if context.recorder else 0.0
    verdict = None
    try:
        result = await asyncio.wait_for(
            Runner.run(roster.clarifier, question, session=None, context=context, max_turns=1),
            timeout=config.CLARIFIER_TIMEOUT_S,
        )
        verdict = result.final_output
    except Exception:
        pass
    if context.recorder:
        context.recorder.note(
            name="clarify", agent="Clarifier", args={"question": question},
            result=(verdict.model_dump_json() if isinstance(verdict, Clarification)
                    else "gate failed or timed out; routed to the full pipeline"),
            started_at=started, ended_at=context.recorder.elapsed(),
        )
    if not isinstance(verdict, Clarification):
        return Verdict()
    if verdict.ready or not verdict.question.strip():
        return Verdict(simple=verdict.simple)

    # Out of scope is a statement of fact, not a question, so it gets no marker: there is nothing
    # to answer, and it must not consume this conversation's one clarification.
    if verdict.out_of_scope:
        return Verdict(reply=verdict.question.strip())

    reply = CLARIFY_MARK + verdict.question.strip()
    if session is not None:
        # Written by hand because no Runner.run touched the session: the next turn must see both
        # the vague question and what we asked, or it answers an orphaned reply.
        await session.add_items([
            {"role": "user", "content": question},
            {"role": "assistant", "content": reply},
        ])
    return Verdict(reply=reply)


async def _already_asked(session: Session | None) -> bool:
    """Whether this conversation already contains a clarifying question of ours.

    Matched on an assistant message that *starts* with the marker. Matching it anywhere would let a
    user disable the gate for the rest of the conversation by quoting the phrase back.
    """
    if session is None:
        return False
    try:
        items = await session.get_items()
    except Exception:
        return False
    return any(item.get("role") == "assistant" and str(item.get("content", "")).startswith(CLARIFY_MARK)
               for item in items)


async def _retry_as_analyst(question: str, session: Session | None, model_id: str,
                            recorder: ToolRecorder, context: RunContext) -> str:
    """Answer on the fast path after the Lead reached for a tool it does not have."""
    try:
        result = await asyncio.wait_for(
            Runner.run(roster.analyst.clone(model=roster.model(model_id)), question,
                       session=session, hooks=recorder, context=context, max_turns=config.MAX_TURNS),
            timeout=max(15.0, context.seconds_left()),
        )
        return _attest(str(result.final_output or "").strip(), recorder.tool_calls, question)
    except Exception:
        return ("I could not answer that one. Try naming the symbol or sector explicitly, "
                "for example \"a price table for NVDA over the last year\".")


async def _salvage(question: str, context: RunContext) -> str:
    """Write the report the Lead ran out of turns before requesting.

    A Lead that burns its turns has usually already done the research, so raising here would
    discard findings that cost real time and tokens.
    """
    if not context.findings:
        return OUT_OF_TURNS
    try:
        return await research.write_report_now(question, context)
    except Exception:
        return OUT_OF_TURNS
