"""The fan-out: one Lead tool call starts several Researchers at once.

Why a tool that owns `asyncio.gather`, rather than `Agent.as_tool()` or handoffs:

- **The requirement is that several researchers run in one turn.** With `as_tool` that depends on
  the model choosing to emit N parallel tool calls over the LiteLLM -> Vertex path, which is not
  reliable on a Flash-Lite class model. With `gather` it is a property of our own code. A graded
  requirement should not be delegated to the weakest model in the lineup.
- **Attribution.** `on_tool_end` receives the agent object, so two concurrent runs of one shared
  Researcher are indistinguishable and the transcript cannot show who did what. Cloning per brief
  with a distinct name fixes that, and the clone costs nothing.
- **Failure isolation.** Sibling tasks are cancelled when one raises. `return_exceptions=True` plus
  a per-brief timeout is what keeps one stuck researcher from losing the other three.

Handoffs are disqualified outright: they transfer control to one agent and pass the whole history,
so there is no parallelism and no fresh context.

Each researcher gets `session=None` - a clean context holding only its brief. That is the point of
fanning out: four small focused contexts rather than one that has read everything.
"""

import asyncio
import time
import uuid

from agents import (
    Agent,
    MaxTurnsExceeded,
    RunContextWrapper,
    Runner,
    SQLiteSession,
    function_tool,
)

import citations
import config
import roster
from run_context import RunContext
from schemas import Report, ResearcherReport

OVER_ROUNDS = (
    "You have used all {rounds} research rounds. Call write_report now with what you have; "
    "do not start another round."
)

OVER_BUDGET = (
    "The time budget for this question is spent. Call write_report now with the findings already "
    "gathered, and say in the caveats that the research was cut short."
)


def _brief_agent(brief: str, n: int, model_id: str) -> Agent:
    """A Researcher named for its brief, so the transcript can tell concurrent runs apart."""
    # The name is what on_tool_end reports, so it has to identify the brief, not just the role.
    label = brief.strip().split("\n")[0][:48]
    return roster.researcher.clone(name=f"Researcher {n}: {label}", model=roster.model(model_id))


async def _report_now(agent: Agent, memory: SQLiteSession, ctx: RunContext) -> ResearcherReport | None:
    """Ask a researcher that ran out of road to write up what it already has.

    Its own tool results are still in `memory`, so the work is recoverable - and the first live run
    of this pipeline showed why that matters: both researchers spent their whole budget gathering
    and returned nothing, so a 30-second run produced no evidence at all. The clone has **no
    tools**, which removes the option of gathering more and forces the structured report out.
    """
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
            timeout=min(30.0, max(5.0, ctx.seconds_left())),
        )
        return result.final_output if isinstance(result.final_output, ResearcherReport) else None
    except Exception:
        return None


async def _one(brief: str, n: int, ctx: RunContext) -> dict:
    """Run one brief to completion and return its report as a plain dict.

    Never raises: a researcher that fails, times out or returns nothing becomes a recorded gap, so
    the Lead can see what is missing instead of the whole round collapsing.
    """
    agent = _brief_agent(brief, n, ctx.model_id)
    # Private and in-memory: a clean context holding only this brief, which is the point of fanning
    # out, while still being a real history we can come back to if the turn budget runs out.
    memory = SQLiteSession(f"brief-{n}-{uuid.uuid4().hex[:8]}")
    started = round(time.monotonic() - ctx.recorder._origin, 3)
    outcome, report = "ok", None
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
            timeout=min(config.RESEARCHER_TIMEOUT_S, max(5.0, ctx.seconds_left())),
        )
        report = result.final_output
    except (asyncio.TimeoutError, MaxTurnsExceeded) as e:
        # Out of time or out of turns, but the tool results are in `memory`. Salvage them.
        limit = ("the time limit" if isinstance(e, asyncio.TimeoutError)
                 else f"all {config.MAX_RESEARCHER_TURNS} turns")
        report = await _report_now(agent, memory, ctx)
        outcome = (f"hit {limit}; reported from partial work" if report
                   else f"hit {limit} with nothing usable gathered")
    except Exception as e:  # a researcher failing must not take the round down with it
        outcome = f"failed: {type(e).__name__}: {str(e)[:120]}"
    finally:
        memory.close()
    ended = round(time.monotonic() - ctx.recorder._origin, 3)

    # One synthetic entry per researcher: the SDK reports the Lead's single `research` call, so
    # without this the transcript would show four researchers as one step with no overlap visible.
    ctx.recorder.note(
        name="researcher",
        agent=agent.name,
        args={"brief": brief},
        result=(report.summary if isinstance(report, ResearcherReport) else outcome),
        started_at=started,
        ended_at=ended,
    )

    if not isinstance(report, ResearcherReport):
        return {"brief": brief, "findings": [], "summary": outcome, "gap": outcome}
    return {
        "brief": brief,
        "findings": [f.model_dump() for f in report.findings],
        "summary": report.summary,
        # A salvaged report is still partial, so say so even when the researcher claimed no gap.
        "gap": report.gap or ("" if outcome == "ok" else outcome),
    }


async def run_round(briefs: list[str], ctx: RunContext) -> list[dict]:
    """Run every brief concurrently and return their reports, in brief order.

    `return_exceptions=True` is belt and braces: `_one` already swallows its own failures, but a
    bug there must still not cancel the siblings.
    """
    limit = asyncio.Semaphore(config.MAX_CONCURRENT_RESEARCHERS)

    async def guarded(brief: str, n: int) -> dict:
        async with limit:
            return await _one(brief, n, ctx)

    results = await asyncio.gather(
        *(guarded(b, i + 1) for i, b in enumerate(briefs)), return_exceptions=True
    )
    return [
        r if isinstance(r, dict)
        else {"brief": briefs[i], "findings": [], "summary": f"failed: {r}", "gap": str(r)}
        for i, r in enumerate(results)
    ]


def _store(round_no: int, reports: list[dict], ctx: RunContext) -> list[dict]:
    """Give every finding an id and keep it. Ids are minted here, never by a model.

    `F{round}.{brief}.{k}` is stable and collision-free across concurrent researchers, which is
    exactly what a model cannot guarantee for itself.
    """
    cited = []
    for b, report in enumerate(reports, start=1):
        for k, finding in enumerate(report["findings"], start=1):
            fid = f"F{round_no}.{b}.{k}"
            ctx.findings[fid] = {"id": fid, **finding}
            cited.append(ctx.findings[fid])
        if report["gap"]:
            ctx.gaps.append(report["gap"])
    return cited


@function_tool
async def research(wrapper: RunContextWrapper[RunContext], briefs: list[str]) -> str:
    """Send 2-4 research briefs to separate researchers, who work on them at the same time.

    Each researcher starts with a clean context and sees only its own brief, so a brief must be
    self-contained: name the symbols, sector or ETF family and say exactly what to establish. Split
    the question by subject rather than by tool - 'screen the energy sector' and 'explain the DVN
    divergence' are two briefs; 'call screen_pairs' and 'then call analyze_pair' are not.

    Returns every finding with the id to cite it by.

    Args:
        briefs: Two to four self-contained research tasks, each one or two sentences.
    """
    ctx = wrapper.context
    if ctx.rounds_used >= config.MAX_FANOUT_ROUNDS:
        return OVER_ROUNDS.format(rounds=config.MAX_FANOUT_ROUNDS)
    if ctx.over_budget():
        return OVER_BUDGET

    briefs = [b.strip() for b in briefs if b and b.strip()][: config.MAX_BRIEFS_PER_ROUND]
    if not briefs:
        return "No briefs were given. Send two to four self-contained research tasks."

    ctx.rounds_used += 1
    reports = await run_round(briefs, ctx)
    cited = _store(ctx.rounds_used, reports, ctx)

    lines = [f"Round {ctx.rounds_used} of {config.MAX_FANOUT_ROUNDS}: {len(briefs)} researchers ran in parallel."]
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
    rounds_left = config.MAX_FANOUT_ROUNDS - ctx.rounds_used
    lines.append(
        f"\n{rounds_left} round(s) left. Call research again only to fill a stated gap; "
        f"otherwise call write_report now."
    )
    return "\n".join(lines)


async def _write(question: str, ctx: RunContext) -> str:
    """Run the writer on the evidence gathered so far, then audit what it wrote.

    The writer has no tools, so the only numbers available to it are the findings listed here. The
    audit is the backstop for the rest: a citation pointing at an id that was never minted, or a
    sentence stating a number with no citation at all. One retry, then any remaining claim is
    marked rather than deleted.
    """
    if not ctx.findings:
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

    agent = roster.writer.clone(model=roster.model(ctx.model_id))
    report, answer = None, ""
    for attempt in (1, 2):
        try:
            result = await asyncio.wait_for(
                Runner.run(agent, prompt, session=None, hooks=ctx.recorder, context=ctx, max_turns=2),
                timeout=min(config.WRITER_TIMEOUT_S, max(5.0, ctx.seconds_left())),
            )
        except Exception as e:
            return f"The writer could not finish ({type(e).__name__}). Findings gathered:\n\n{evidence}"
        report = result.final_output
        answer = (report.answer if isinstance(report, Report) else str(report or "")).strip()
        checked = citations.audit(answer, ctx.findings)
        if checked["clean"] or attempt == 2 or ctx.over_budget():
            break
        prompt = f"{prompt}\n\nYOUR DRAFT\n{answer}\n\n{citations.feedback(checked)}"

    checked = citations.audit(answer, ctx.findings)
    if not checked["clean"]:
        answer = citations.mark_unsupported(answer, checked)
        checked = citations.audit(answer, ctx.findings)
    caveats = report.caveats.strip() if isinstance(report, Report) and report.caveats else ""
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
        question: The user's question, restated in one sentence, for the writer to answer.
    """
    wrapper.context.report = await _write(question, wrapper.context)
    return "The report is written and has been returned to the user. Stop now; say nothing further."


def lead_agent(model_id: str) -> Agent:
    """The lead, wired to its only two tools. Built per run so it uses the caller's model."""
    return roster.lead.clone(model=roster.model(model_id), tools=[research, write_report])
