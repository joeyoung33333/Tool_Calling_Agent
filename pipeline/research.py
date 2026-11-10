"""The fan-out: one Lead tool call starts several Researchers at once.

Why a tool that owns `asyncio.gather`, rather than `Agent.as_tool()` or handoffs:

- **Parallelism is ours, not the model's.** With `as_tool` it depends on the model emitting N
  parallel tool calls over the LiteLLM -> Vertex path, which a Flash-Lite class model does not do
  reliably. With `gather` it is a property of our own code.
- **Attribution.** `on_tool_end` receives the agent object, so two concurrent runs of one shared
  Researcher are indistinguishable and the transcript cannot show who did what. Cloning per brief
  with a distinct name fixes that, and the clone costs nothing.
- **Failure isolation.** Sibling tasks are cancelled when one raises, so `return_exceptions=True`
  plus a per-brief timeout is what keeps one stuck researcher from losing the other three.

Handoffs are disqualified outright: they transfer control to one agent and pass the whole history,
so there is neither parallelism nor a fresh context.

Each researcher gets its own private session - a clean context holding only its brief, which is the
point of fanning out: several small focused contexts rather than one that has read everything.
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
    function_tool,
)

import config
from pipeline import citations, roster
from pipeline.runtime import RunContext
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
    """A Researcher named for its brief, so the transcript can tell concurrent runs apart.

    The name is what `on_tool_end` reports, so it must identify the brief and not just the role.
    """
    label = brief.strip().split("\n")[0][:48]
    return roster.researcher.clone(name=f"Researcher {n}: {label}", model=roster.model(model_id))


async def _report_now(agent: Agent, memory: SQLiteSession, ctx: RunContext) -> ResearcherReport | None:
    """Ask a researcher that ran out of road to write up what it already has.

    Its tool results are still in `memory`, so the work is recoverable. The clone has **no tools**,
    which removes the option of gathering more and forces the structured report out.
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
    the Lead sees what is missing instead of the round collapsing.
    """
    agent = _brief_agent(brief, n, ctx.model_id)
    # Private and in-memory: only this brief, but still a real history to salvage from if the turn
    # budget runs out.
    memory = SQLiteSession(f"brief-{n}-{uuid.uuid4().hex[:8]}")
    started = ctx.recorder.elapsed()
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
        # Out of time or turns, but the tool results are in `memory`, so salvage them.
        limit = ("the time limit" if isinstance(e, asyncio.TimeoutError)
                 else f"all {config.MAX_RESEARCHER_TURNS} turns")
        report = await _report_now(agent, memory, ctx)
        outcome = (f"hit {limit}; reported from partial work" if report
                   else f"hit {limit} with nothing usable gathered")
    except litellm.RateLimitError:
        ctx.note_rate_limit()
        outcome = "the model rate limited this brief (HTTP 429)"
    except Exception as e:  # a researcher failing must not take the round down with it
        outcome = f"failed: {type(e).__name__}: {str(e)[:120]}"
    finally:
        memory.close()
    ended = ctx.recorder.elapsed()

    # One synthetic entry per researcher. The SDK reports only the Lead's single `research` call,
    # so without this the transcript would show the whole fan-out as one step.
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

    `return_exceptions=True` is belt and braces: `_one` swallows its own failures, but a bug there
    must still not cancel the siblings.
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


def _sources(ctx: RunContext) -> dict[str, str]:
    """tool name -> the `source` its results reported this run.

    A finding names the tool that produced it, which answers "who computed this" but not "where did
    the data come from". Every tool result already states that; this reads it back off the
    transcript so the evidence list can show the feed behind each number.
    """
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
    """Give every finding an id and keep it. Ids are minted here, never by a model.

    `F{round}.{brief}.{k}` is collision-free across concurrent researchers, which is exactly what a
    model cannot guarantee for itself.
    """
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
    if ctx.rounds_used >= config.MAX_FANOUT_ROUNDS:
        return OVER_ROUNDS.format(rounds=config.MAX_FANOUT_ROUNDS)
    if ctx.over_budget():
        return OVER_BUDGET

    briefs = [b.strip() for b in briefs if b and b.strip()][: config.MAX_BRIEFS_PER_ROUND]
    if not briefs:
        return "No briefs were given. Send two to four self-contained research tasks."

    if ctx.throttled():
        return THROTTLED.format(n=ctx.rate_limited)

    ctx.rounds_used += 1
    reports = await run_round(briefs, ctx)
    if ctx.throttled():
        lines_after = THROTTLED.format(n=ctx.rate_limited)
    else:
        lines_after = ""
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
    """Run the writer on the evidence gathered so far, then audit what it wrote.

    The writer has no tools, so the only numbers available to it are the findings listed here. One
    retry on a failed audit, then any remaining uncited claim is marked rather than deleted.
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
    # The writer holds no tools, so the SDK reports nothing for it and the transcript would end at
    # the Lead's write_report call. Recorded by hand, as the Clarifier is.
    started = ctx.recorder.elapsed() if ctx.recorder else 0.0
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

    # The caveats are printed under the answer, so an uncited number there would reach the reader
    # without passing the check the answer passes. Audited on the same terms, and marked not deleted.
    if ctx.recorder:
        ctx.recorder.note(
            name="writer", agent="Writer",
            args={"findings": len(ctx.findings), "gaps": len(ctx.gaps)},
            result=f"wrote {len(answer)} characters citing {len(checked['cited'])} of "
                   f"{len(ctx.findings)} findings",
            started_at=started, ended_at=ctx.recorder.elapsed(),
        )

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
    wrapper.context.report = await write_report_now(question, wrapper.context)
    return "The report is written and has been returned to the user. Stop now; say nothing further."


def lead_agent(model_id: str) -> Agent:
    """The lead, wired to its only two tools. Built per run so it uses the caller's model."""
    return roster.lead.clone(model=roster.model(model_id), tools=[research, write_report])
