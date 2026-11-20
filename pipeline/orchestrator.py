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

TIMED_OUT_EMPTY = (
    "**This question ran out of time** before anything usable was gathered. A narrower question "
    "will finish."
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
    top_name = "Analyst" if verdict.simple else "Lead"
    run = await _run_top(agent, question, session, recorder, context, verdict.simple, budget)
    _note_top(recorder, top_name, question, verdict.simple, context, run)

    # A Lead whose model failed before any research ran is retried once on the fallback model.
    if not verdict.simple and run.model_failed and not context.rounds_used:
        model = roster.model_name(config.FALLBACK_MODEL)
        await _drop_unanswered(session, question)
        run = await _run_top(research.lead_agent(config.FALLBACK_MODEL), question, session, recorder,
                             context, False, max(15.0, context.research_left()))
        _note_top(recorder, "Lead", question, False, context, run, retry_on=model)

    response, timed_out, needs_writer = run.response, run.timed_out, not verdict.simple and not run.retry
    if run.retry:
        retry_started = recorder.elapsed()
        response = await _retry_as_analyst(question, session, model_id, recorder, context)
        recorder.note(name="agent", agent="Analyst",
                      args={"question": question, "route": "fast path", "retry": True},
                      result="retried on the fast path after the Lead failed",
                      started_at=retry_started, ended_at=recorder.elapsed())

    if needs_writer and (run.ended_normally or context.findings):
        response = await research.write_report_now(context.report_question or question, context)
        if session is not None and response:
            # The writer runs outside the Lead's run, so save its answer by hand for the next turn.
            await session.add_items([{"role": "assistant", "content": response}])
    if timed_out:
        response = OVER_TIME + "\n\n" + response if response else TIMED_OUT_EMPTY

    return {
        "response": response or (NOTHING_FOUND if not context.findings else EMPTY),
        "tool_calls": recorder.tool_calls,
        "charts": context.charts,
    }


@dataclass
class TopRun:
    """How the Analyst's or the Lead's run ended."""

    started_at: float
    response: str = ""
    error: str = ""
    ended_normally: bool = False
    timed_out: bool = False
    model_failed: bool = False  # the model call itself failed, so another model may succeed
    retry: bool = False  # the Lead reached for a data tool, so the Analyst should answer


async def _run_top(agent, question: str, session: Session | None, recorder: ToolRecorder,
                   context: RunContext, simple: bool, budget: float) -> TopRun:
    """Run the Analyst or the Lead within `budget` seconds; failures are returned, not raised."""
    run = TopRun(started_at=recorder.elapsed())
    try:
        result = await asyncio.wait_for(
            Runner.run(agent, question, session=session, hooks=recorder, context=context,
                       max_turns=config.MAX_TURNS if simple else config.LEAD_TURNS),
            timeout=budget,
        )
        run.ended_normally = True
        if simple:
            run.response = _attest(str(result.final_output or "").strip(), recorder.tool_calls, question)
    except MaxTurnsExceeded:
        run.error, run.response = "ran out of turns", OUT_OF_TURNS  # findings are still written up
    except ModelBehaviorError as e:
        if simple:
            run.error = type(e).__name__
            run.response = f"The model call failed ({run.error}); the tool calls below show what was tried."
        else:
            # The Lead imitated a data tool from the session history; the analyst holds that tool.
            run.retry, run.error = True, "called a tool it does not have; retried as the Analyst"
    except asyncio.TimeoutError:
        run.timed_out, run.error = True, "timed out"
    except litellm.RateLimitError:
        run.error, run.response, run.model_failed = "rate limited", RATE_LIMITED, True
    except Exception as e:
        # Any other model failure: findings already gathered are still written up.
        run.error, run.model_failed = type(e).__name__, True
        run.response = f"The model call failed ({run.error}); the tool calls below show what was tried."
    return run


def _note_top(recorder: ToolRecorder, name: str, question: str, simple: bool, context: RunContext,
              run: TopRun, retry_on: str = "") -> None:
    """One timeline row for the Analyst's or the Lead's run; red when it failed."""
    args = {"question": question, "route": "fast path" if simple else "research path"}
    if retry_on:
        args.update(model=retry_on, retry=True)
    recorder.note(
        name="agent", agent=name, args=args,
        result=(f"{run.error}; " if run.error else "")
               + f"{len(recorder.tool_calls)} tool calls, {len(context.findings)} findings",
        started_at=run.started_at, ended_at=recorder.elapsed(), error=run.error,
    )


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


async def _drop_unanswered(session: Session | None, question: str) -> None:
    """Remove the question a failed run saved, so its retry does not store it a second time."""
    if session is None:
        return
    last = await session.get_items(limit=1)
    if last and last[-1].get("role") == "user" and last[-1].get("content") == question:
        await session.pop_item()


async def _retry_as_analyst(question: str, session: Session | None, model_id: str,
                            recorder: ToolRecorder, context: RunContext) -> str:
    """Answer on the fast path after the Lead reached for a tool it does not have."""
    try:
        result = await asyncio.wait_for(
            Runner.run(roster.analyst.clone(model=roster.model(model_id)), question,
                       session=session, hooks=recorder, context=context, max_turns=config.MAX_TURNS),
            timeout=max(5.0, context.seconds_left()),
        )
        return _attest(str(result.final_output or "").strip(), recorder.tool_calls, question)
    except Exception:
        return ("I could not answer that one. Try naming the symbol or sector explicitly, "
                "for example \"a price table for NVDA over the last year\".")
