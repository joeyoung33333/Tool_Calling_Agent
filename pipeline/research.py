"""The fan-out: one Lead tool call runs several Researchers concurrently with asyncio.gather.

gather, not as_tool or handoffs, keeps parallelism in our code; each researcher sees only its brief.
"""

import asyncio
import json
import uuid

import litellm

from agents import (
    Agent,
    MaxTurnsExceeded,
    RunContextWrapper,
    Runner,
    SQLiteSession,
    StopAtTools,
    function_tool,
)

import config
from pipeline import citations, roster
from pipeline.runtime import RunContext, ToolRecorder
from pipeline.schemas import Report, ResearcherReport

OVER_ROUNDS = (
    "You have used all {rounds} research rounds. Call write_report now with what you have; "
    "do not start another round."
)

THROTTLED = (
    "The data and model providers have refused {n} requests in this run (HTTP 429), so more "
    "research would be refused too. Call write_report now with what you have, and say in the "
    "restated question that the research was cut short by rate limits."
)

OVER_BUDGET = (
    "The time budget for this question is spent. Call write_report now, and say in the restated "
    "question that the research was cut short so the writer reports it."
)


def _brief_agent(brief: str, n: int, model_id: str) -> Agent:
    """A Researcher named for its brief: the name is what `on_tool_end` reports."""
    label = brief.strip().split("\n")[0][:48]
    return roster.researcher.clone(name=f"Researcher {n}: {label}", model=roster.model(model_id))


async def _report_now(agent: Agent, memory: SQLiteSession, ctx: RunContext) -> ResearcherReport | None:
    """Salvage a stalled researcher: a tool-less clone writes up what is already in `memory`."""
    try:
        result = await asyncio.wait_for(
            Runner.run(
                agent.clone(tools=[]),
                "Stop researching. Report what you have already found, using only the tool results "
                "above. If you found nothing usable, return no findings and say so in `gap`.",
                session=memory,
                hooks=ctx.recorder,
                context=ctx,
                max_turns=2,
            ),
            timeout=min(config.RESEARCHER_SALVAGE_S, max(1.0, ctx.research_left() - config.ROUND_MARGIN_S)),
        )
        return result.final_output if isinstance(result.final_output, ResearcherReport) else None
    except Exception:
        return None


async def _one(brief: str, n: int, ctx: RunContext) -> dict:
    """Run one brief and return its report as a dict. Never raises: failures become a gap."""
    agent = _brief_agent(brief, n, ctx.model_id)
    # Private and in-memory, but a real history to salvage from.
    memory = SQLiteSession(f"brief-{n}-{uuid.uuid4().hex[:8]}")
    started = ctx.recorder.elapsed()
    outcome, report, error, salvage_started = "ok", None, "", None
    try:
        result = await asyncio.wait_for(
            Runner.run(
                agent,
                brief,
                session=memory,
                hooks=ctx.recorder,  # nested calls land in the same transcript
                context=ctx,  # so its tools can still draw to the page
                max_turns=config.MAX_RESEARCHER_TURNS,
            ),
            # Leaves room for the salvage write-up before the main run is cut off.
            timeout=min(config.RESEARCHER_TIMEOUT_S, max(
                5.0, ctx.research_left() - config.RESEARCHER_SALVAGE_S - config.ROUND_MARGIN_S)),
        )
        report = result.final_output
    except (asyncio.TimeoutError, MaxTurnsExceeded) as e:
        limit = ("the time limit" if isinstance(e, asyncio.TimeoutError)
                 else f"all {config.MAX_RESEARCHER_TURNS} turns")
        error = "timed out" if isinstance(e, asyncio.TimeoutError) else "ran out of turns"
        salvage_started = ctx.recorder.elapsed()
        report = await _report_now(agent, memory, ctx)
        outcome = (f"hit {limit}; reported from partial work" if report
                   else f"hit {limit} with nothing usable gathered")
    except litellm.RateLimitError:
        ctx.note_rate_limit()
        error, outcome = "rate limited", "the model rate limited this brief (HTTP 429)"
    except Exception as e:  # a researcher failing must not take the round down with it
        error, outcome = type(e).__name__, f"failed: {type(e).__name__}: {str(e)[:120]}"
    finally:
        memory.close()
    ended = ctx.recorder.elapsed()

    # Recorded by hand (the SDK sees only `research`); a stalled one is a red row plus its write-up.
    summary = report.summary if isinstance(report, ResearcherReport) else outcome
    if salvage_started is None:
        ctx.recorder.note(name="researcher", agent=agent.name, args={"brief": brief}, result=summary,
                          started_at=started, ended_at=ended, error=error)
    else:
        ctx.recorder.note(name="researcher", agent=agent.name, args={"brief": brief},
                          result=f"{error}; writing up what it had gathered",
                          started_at=started, ended_at=salvage_started, error=error)
        ctx.recorder.note(name="researcher", agent=agent.name,
                          args={"brief": brief, "retry": True, "note": "wrote up its partial work, no tools"},
                          result=summary, started_at=salvage_started, ended_at=ended,
                          error="" if report else "nothing usable gathered")

    if not isinstance(report, ResearcherReport):
        return {"brief": brief, "findings": [], "summary": outcome, "gap": outcome}
    return {
        "brief": brief,
        "findings": [f.model_dump() for f in report.findings],
        "summary": report.summary,
        # A salvaged report is still partial, so say so even when the researcher claimed no gap.
        "gap": report.gap or ("" if outcome == "ok" else outcome),
    }


async def run_round(briefs: list[str], ctx: RunContext, done: dict | None = None) -> list[dict]:
    """Run every brief concurrently and return their reports, in brief order.

    `done` collects reports as they finish, so a cancelled caller can keep them.
    """
    limit = asyncio.Semaphore(config.MAX_CONCURRENT_RESEARCHERS)

    async def guarded(brief: str, n: int) -> dict:
        async with limit:
            report = await _one(brief, n, ctx)
        if done is not None:
            done[n] = report
        return report

    results = await asyncio.gather(
        *(guarded(b, i + 1) for i, b in enumerate(briefs)), return_exceptions=True
    )
    return [
        r if isinstance(r, dict)
        else {"brief": briefs[i], "findings": [], "summary": f"failed: {r}", "gap": str(r)}
        for i, r in enumerate(results)
    ]


def _sources(ctx: RunContext) -> dict[str, str]:
    """tool name -> the `source` its results reported this run, for the evidence list."""
    found: dict[str, str] = {}
    for call in (getattr(ctx.recorder, "tool_calls", None) or []):
        if call["name"] in found:
            continue
        try:
            got = json.loads(str(call.get("result", "")))
        except (TypeError, ValueError):
            continue
        if isinstance(got, dict) and got.get("source"):
            found[call["name"]] = str(got["source"])
    return found


def _store(round_no: int, reports: list[dict], ctx: RunContext) -> list[dict]:
    """Give every finding an id and keep it. Ids are minted here, never by a model."""
    sources = _sources(ctx)
    cited = []
    for b, report in enumerate(reports, start=1):
        for k, finding in enumerate(report["findings"], start=1):
            fid = f"F{round_no}.{b}.{k}"
            ctx.findings[fid] = {"id": fid, "source": sources.get(finding.get("tool", ""), ""), **finding}
            cited.append(ctx.findings[fid])
        if report["gap"]:
            ctx.gaps.append(report["gap"])
    return cited


@function_tool
async def research(wrapper: RunContextWrapper[RunContext], briefs: list[str]) -> str:
    """Send 2-4 research briefs to separate researchers, who work on them at the same time.

    Each researcher sees only its own brief, so each must be self-contained: name the symbols,
    sector or ETF family and say what to establish. Returns every finding with the id to cite it by.

    Args:
        briefs: Two to four self-contained research tasks, each one or two sentences.
    """
    ctx = wrapper.context
    # A refusal is recorded as a failed step (ToolRecorder reads ctx.refused).
    if ctx.rounds_used >= config.MAX_FANOUT_ROUNDS:
        ctx.refused = "round limit reached"
        return OVER_ROUNDS.format(rounds=config.MAX_FANOUT_ROUNDS)
    if ctx.research_left() <= config.RESEARCHER_SALVAGE_S + config.MIN_RESEARCHER_S:
        ctx.refused = "not enough time left to start a round"
        return OVER_BUDGET

    briefs = [b.strip() for b in briefs if b and b.strip()][: config.MAX_BRIEFS_PER_ROUND]
    if len(briefs) < 2:
        ctx.refused = "fewer than two briefs"
        return ("Send two to four self-contained briefs in one call: split the question by subject "
                "so researchers can work on it at the same time.")

    if ctx.throttled():
        ctx.refused = "rate limited"
        return THROTTLED.format(n=ctx.rate_limited)

    # Taken before the await: concurrent research calls must not share round ids.
    ctx.rounds_used += 1
    round_no = ctx.rounds_used
    done: dict[int, dict] = {}
    try:
        reports = await run_round(briefs, ctx, done)
    except asyncio.CancelledError:
        # Cut off mid-round: keep what finished for the writer.
        cut = "cut off by the time limit before reporting"
        _store(round_no, [done.get(i + 1) or {"brief": b, "findings": [], "summary": cut, "gap": cut}
                          for i, b in enumerate(briefs)], ctx)
        raise
    lines_after = THROTTLED.format(n=ctx.rate_limited) if ctx.throttled() else ""
    cited = _store(round_no, reports, ctx)

    lines = [f"Round {round_no} of {config.MAX_FANOUT_ROUNDS}: {len(briefs)} researchers ran in parallel."]
    for report in reports:
        lines.append(f"\nBRIEF: {report['brief']}\n  {report['summary']}")
        if report["gap"]:
            lines.append(f"  GAP: {report['gap']}")
    lines.append("\nEVIDENCE (cite by id; the writer can only use these):")
    for f in cited:
        where = f" [{f['pair']}]" if f.get("pair") else ""
        lines.append(f"  {f['id']}{where} {f['metric']} = {f['value']} ({f['tool']}) - {f['claim']}")
    if not cited:
        lines.append("  none - no researcher produced a usable finding.")
    if lines_after:
        lines.append("\n" + lines_after)
        return "\n".join(lines)
    rounds_left = config.MAX_FANOUT_ROUNDS - ctx.rounds_used
    lines.append(
        f"\n{rounds_left} round(s) left. Call research again only to fill a stated gap; "
        f"otherwise call write_report now."
        if rounds_left else "\nNo rounds left. Call write_report now with what you have."
    )
    return "\n".join(lines)


async def write_report_now(question: str, ctx: RunContext) -> str:
    """Write up the evidence so far: a failed draft is retried on the fallback model, a draft that
    fails the citation audit is rewritten once, and whatever still fails is marked."""
    if not ctx.findings:
        if not ctx.rounds_used:
            return ("The research never started: planning the question used up the time budget "
                    "before any researcher ran, so there is nothing to report. Please try again.")
        return (
            "No researcher produced a usable finding for this question, so there is nothing to "
            "report. "
            + (" ".join(ctx.gaps[:3]) if ctx.gaps else "The tools returned no evidence.")
        )

    evidence = "\n".join(
        f"[{f['id']}]{' ' + f['pair'] if f.get('pair') else ''} {f['metric']} = {f['value']} "
        f"(via {f['tool']}) - {f['claim']}"
        + (f" LIMITATION: {f['limitation']}" if f.get("limitation") else "")
        for f in ctx.findings.values()
    )
    gaps = ("\n\nGAPS the researchers reported:\n" + "\n".join(f"- {g}" for g in ctx.gaps)) if ctx.gaps else ""
    prompt = f"QUESTION\n{question}\n\nFINDINGS (the only facts available to you)\n{evidence}{gaps}"

    # The writer has no tools, so the SDK never reports it; each attempt is recorded by hand.
    rec = ctx.recorder or ToolRecorder()
    args = {"findings": len(ctx.findings), "gaps": len(ctx.gaps)}

    async def draft(model_id: str, text: str, keep_back: float):
        result = await asyncio.wait_for(
            Runner.run(roster.writer.clone(model=roster.model(model_id)), text, session=None,
                       hooks=ctx.recorder, context=ctx, max_turns=2),
            timeout=min(config.WRITER_TIMEOUT_S, max(5.0, ctx.seconds_left() - keep_back)),
        )
        report = result.final_output
        return report, (report.answer if isinstance(report, Report) else str(report or "")).strip()

    # The pinned model first; a failed draft is retried on the fallback, with time kept back for it.
    attempts = [(config.REASONING_MODEL, config.WRITER_FALLBACK_S), (config.FALLBACK_MODEL, 0.0)]
    report, answer, model_id = None, "", config.REASONING_MODEL
    for i, (model_id, keep_back) in enumerate(attempts):
        started = rec.elapsed()
        try:
            report, answer = await draft(model_id, prompt, keep_back)
            error = "" if answer else "returned an empty answer"
        except Exception as e:
            error = "timed out" if isinstance(e, asyncio.TimeoutError) else type(e).__name__
        if not error:
            break
        last = i == len(attempts) - 1
        rec.note(name="writer", agent="Writer",
                 args={**args, "model": roster.model_name(model_id), **({"retry": True} if i else {})},
                 result=f"{error} on {roster.model_name(model_id)}; "
                        + ("no answer written" if last
                           else f"retrying on {roster.model_name(attempts[i + 1][0])}"),
                 started_at=started, ended_at=rec.elapsed(), error=error)
    if not answer:
        return f"The writer could not finish. Findings gathered:\n\n{evidence}"

    # One rewrite against the citation audit; if it fails, the draft is kept and marked below.
    checked = citations.audit(answer, ctx.findings)
    if not checked["clean"] and not ctx.over_budget():
        try:
            redone, rewritten = await draft(
                model_id, f"{prompt}\n\nYOUR DRAFT\n{answer}\n\n{citations.feedback(checked)}", 0.0)
            if rewritten:
                report, answer = redone, rewritten
        except Exception:
            pass

    checked = citations.audit(answer, ctx.findings)
    if not checked["clean"]:
        answer = citations.mark_unsupported(answer, checked)
        checked = citations.audit(answer, ctx.findings)

    rec.note(name="writer", agent="Writer",
             args={**args, "model": roster.model_name(model_id),
                   **({"retry": True} if model_id != config.REASONING_MODEL else {})},
             result=f"wrote {len(answer)} characters citing {len(checked['cited'])} of "
                    f"{len(ctx.findings)} findings",
             started_at=started, ended_at=rec.elapsed())

    # Caveats reach the reader too, so they get the same audit, marked not deleted.
    caveats = report.caveats.strip() if isinstance(report, Report) and report.caveats else ""
    if caveats:
        caveat_check = citations.audit(caveats, ctx.findings)
        if not caveat_check["clean"]:
            caveats = citations.mark_unsupported(caveats, caveat_check)
    return "\n".join(
        part for part in (
            answer,
            f"\n{caveats}" if caveats else "",
            citations.evidence_block(ctx.findings, checked["cited"]),
        ) if part
    )


@function_tool
async def write_report(wrapper: RunContextWrapper[RunContext], question: str) -> str:
    """Hand the evidence to the writer and finish. Always end a question with this.

    Args:
        question: The user's question, restated in one sentence. This is all the writer sees of
            what was asked, so keep every constraint that matters: the universe, the window, and
            what "good" was taken to mean.
    """
    # The writer runs after the Lead's run ends, on its own reserve (orchestrator.answer).
    wrapper.context.report_question = question
    return "The writer has the evidence and is writing the answer."


def lead_agent() -> Agent:
    """The Lead with its two tools, on the pinned model; the run stops at `write_report`."""
    return roster.lead.clone(model=roster.model(config.REASONING_MODEL), tools=[research, write_report],
                             tool_use_behavior=StopAtTools(stop_at_tool_names=["write_report"]))
