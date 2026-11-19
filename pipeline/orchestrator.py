"""One question in, one answer out: the Clarifier gate, then an analyst or the research pipeline.

The Writer holds no tools, so it can only use findings Python hands it. Failures degrade, not raise.
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

# No turn count: both the analyst (MAX_TURNS) and the Lead (LEAD_TURNS) can reach it.
OUT_OF_TURNS = (
    "I hit my turn limit on this question and could not assemble an answer I can stand behind. "
    "The tool calls below show what was tried. Try a narrower question."
)

# Prefixed onto every clarifying reply; checked in Python so the gate can never loop.
CLARIFY_MARK = "To point the screen at the right names: "

NOTHING_FOUND = (
    "I could not produce an answer I can stand behind: no tool returned a usable result, so there "
    "is nothing I can honestly report. The tool calls below show what was tried."
)

OVER_TIME = (
    "**This question ran out of time**, so the answer below uses only what had been gathered by "
    "then. A narrower question will finish."
)

RATE_LIMITED = (
    "Vertex AI is rate limiting this model (HTTP 429: its per-minute quota is used up). Wait a "
    "minute and try again, or pick a different model."
)


async def answer(question: str, session: Session | None, researcher_model: str) -> dict:
    """Answer one question, continuing `session`; returns /chat's body minus the session id."""
    recorder = ToolRecorder()
    model_id = config.RESEARCHER_MODELS[researcher_model]
    context = RunContext(recorder=recorder, model_id=model_id)

    # Only the Analyst and Researchers follow the dropdown; the rest are pinned.
    verdict = await _clarify(question, session, context)
    if verdict.reply:
        return {"response": verdict.reply, "tool_calls": recorder.tool_calls, "charts": context.charts}

    # A chart or single lookup goes to one analyst; the gate decides.
    agent = (roster.analyst.clone(model=roster.model(model_id)) if verdict.simple
             else research.lead_agent())

    # On the run's clock; research leaves the Writer's reserve, the fast path has no Writer.
    budget = max(15.0, context.seconds_left() if verdict.simple else context.research_left())
    top_started = recorder.elapsed()
    top_name = "Analyst" if verdict.simple else "Lead"
    needs_writer, ended_normally, timed_out, error = not verdict.simple, False, False, ""
    retry = None
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
        ended_normally = True
        response = (_attest(str(result.final_output or "").strip(), recorder.tool_calls, question)
                    if verdict.simple else "")
    except MaxTurnsExceeded:
        error, response = "ran out of turns", OUT_OF_TURNS  # findings are still written up below
    except ModelBehaviorError:
        # The Lead imitated a data tool call from the shared session history; the analyst holds it.
        needs_writer, error, retry = False, "called a tool it does not have; retried as the Analyst", True
    except asyncio.TimeoutError:
        timed_out, error, response = True, "timed out", OUT_OF_TURNS
    except litellm.RateLimitError:
        error, response = "rate limited", RATE_LIMITED
    except Exception as e:
        # Any other model failure: findings already gathered are still written up.
        error = type(e).__name__
        response = f"The model call failed ({type(e).__name__}); the tool calls below show what was tried."

    recorder.note(
        name="agent", agent=top_name,
        args={"question": question, "route": "fast path" if verdict.simple else "research path"},
        result=(f"{error}; " if error else "")
               + f"{len(recorder.tool_calls)} tool calls, {len(context.findings)} findings",
        started_at=top_started, ended_at=recorder.elapsed(), error=error,
    )
    if retry:
        retry_started = recorder.elapsed()
        response = await _retry_as_analyst(question, session, model_id, recorder, context)
        recorder.note(name="agent", agent="Analyst",
                      args={"question": question, "route": "fast path", "retry": True},
                      result="retried on the fast path after the Lead failed",
                      started_at=retry_started, ended_at=recorder.elapsed())

    if needs_writer and (ended_normally or context.findings):
        response = await research.write_report_now(context.report_question or question, context)
        if session is not None and response:
            # The writer runs outside the Lead's run, so save its answer by hand for the next turn.
            await session.add_items([{"role": "assistant", "content": response}])
    if timed_out:
        response = OVER_TIME + "\n\n" + response

    return {
        "response": response or (NOTHING_FOUND if not context.findings else EMPTY),
        "tool_calls": recorder.tool_calls,
        "charts": context.charts,
    }


@dataclass
class Verdict:
    """What the gate decided: answer now, or which way to route."""

    reply: str = ""  # a finished response to return immediately; empty means carry on
    simple: bool = False


def _attest(answer: str, tool_calls: list[dict], question: str = "") -> str:
    """Append the fast path's provenance and name any number found in no tool result."""
    if not answer or not tool_calls:
        return answer
    checked = citations.verify_against_tools(answer, tool_calls, question)
    out = [answer]
    if checked["missing"]:
        out.append(
            "\n**Not found in any tool result from this run, so treat as unverified:** "
            + ", ".join(checked["missing"][:8]) + "."
        )
    out.append(citations.sources_block(tool_calls))
    return "\n".join(part for part in out if part)


async def _clarify(question: str, session: Session | None, context: RunContext) -> Verdict:
    """Decide whether to answer, ask about, or research the question. Failure routes to research."""
    asked = await _already_asked(session)
    # The gate runs without the session, so it is told whether there is history to refer to.
    context.first_turn = not await _has_history(session)
    # The gate has no tools, so the SDK hooks never see it; each attempt is recorded by hand.
    rec = context.recorder or ToolRecorder()
    attempts = [(config.REASONING_MODEL, config.CLARIFIER_TIMEOUT_S),
                (config.FALLBACK_MODEL, config.CLARIFIER_FALLBACK_TIMEOUT_S)]
    verdict = None
    for i, (model_id, timeout) in enumerate(attempts):
        args = {"question": question, "model": roster.model_name(model_id)}
        if i:
            args["retry"] = True
        started = rec.elapsed()
        try:
            result = await asyncio.wait_for(
                Runner.run(roster.clarifier.clone(model=roster.model(model_id)), question,
                           session=None, context=context, max_turns=1),
                timeout=timeout)
            verdict = result.final_output
            error = "" if isinstance(verdict, Clarification) else "no verdict returned"
        except Exception as e:
            error = "timed out" if isinstance(e, asyncio.TimeoutError) else type(e).__name__
        last = i == len(attempts) - 1
        # Each attempt is its own row, so a failed one shows in red with what replaced it.
        rec.note(name="clarify", agent="Clarifier", args=args,
                 result=(verdict.model_dump_json() if not error
                         else f"{error} on {args['model']}; "
                         + ("routed to the full pipeline" if last
                            else f"retrying on {roster.model_name(attempts[i + 1][0])}")),
                 started_at=started, ended_at=rec.elapsed(), error=error)
        if not error:
            break
    if not isinstance(verdict, Clarification):
        return Verdict()
    if verdict.ready or not verdict.question.strip():
        return Verdict(simple=verdict.simple)

    # Out of scope is a statement, not a question: no marker, so it does not use up the one ask.
    if verdict.out_of_scope:
        return Verdict(reply=verdict.question.strip())
    # Never ask twice in one conversation; the gate still picks the route.
    if asked:
        return Verdict(simple=verdict.simple)

    reply = CLARIFY_MARK + verdict.question.strip()
    if session is not None:
        # No Runner.run touched the session, so save both turns or the next answers an orphan.
        await session.add_items([
            {"role": "user", "content": question},
            {"role": "assistant", "content": reply},
        ])
    return Verdict(reply=reply)


async def _already_asked(session: Session | None) -> bool:
    """Whether an assistant message *starts* with our marker (quoting it back must not count)."""
    if session is None:
        return False
    try:
        items = await session.get_items()
    except Exception:
        return False
    return any(item.get("role") == "assistant" and str(item.get("content", "")).startswith(CLARIFY_MARK)
               for item in items)


async def _has_history(session: Session | None) -> bool:
    """Whether this conversation already holds a message. A broken session counts as empty."""
    if session is None:
        return False
    try:
        return bool(await session.get_items())
    except Exception:
        return False


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
