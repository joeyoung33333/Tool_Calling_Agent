# Converge: A Pair-Trading Research Agent

IEOR-E4570: Project 2 — Deep Research

Author: Joseph Young (jay2147)

## The Question

> Given a sector, an ETF family, or a user's own idea, which pairs of US stocks and ETFs are viable
> pair trades — correlated names whose prices move together, drift apart, then converge again? And
> when a pair stays diverged for an unusually long stretch, what news explains it: a temporary
> dislocation, or a relationship that has actually broken?

Converge fits every pair in a universe of up to 40 names, ranks them on in-sample statistics only,
then backtests the best on a holdout the selection never saw. Pairs are chosen by Python, never by
the model: the agent decides *what to investigate*, and the arithmetic decides *what survives*.

Research only, not investment advice.

## Who It's For

Anyone weighing a statistical pair trade who wants the evidence and its limits rather than a
recommendation: analysts, quantitative finance students, and traders checking whether a pair
survives costs, a holdout and a multiple-testing correction. The agent is built to say "the
evidence does not support an answer here" and mean it.

## How It Works

Five agent roles across two routes. The division is the point: each one is denied something, and
what it is denied is what stops it inventing numbers.

```
question
   |
   v
Clarifier   no tools, one turn, structured verdict
   |          vague -> ask one question back and stop
   |          out of scope (crypto, options, FX) -> say so and stop
   |
   +-- simple --> Analyst   all 9 data tools, one agent, ~7s. Every number it
   |                        writes is checked against this run's tool results.
   v
Lead        exactly two tools: research() and write_report(). No data tools at all,
   |        so it cannot state a number it did not receive.
   |
   +-- research(briefs) -------> asyncio.gather
   |                               |-- Researcher 1   all 9 data tools,
   |                               |-- Researcher 2   private context holding
   |                               |-- Researcher 3   one brief, nothing else
   |                               \-- Researcher 4
   |                             each returns structured Findings; Python mints the ids
   |
   +-- (at most one more round, to fill a stated gap)
   |
   v
Writer      NO tools. Sees only the findings it was handed, so the only numbers
   |        it can reach are ones a tool actually produced.
   v
Citation audit (Python)   every number counted against its citation
   v
answer + evidence list
```

**The fan-out is real, and measurable.** `research()` owns `asyncio.gather` rather than using
`Agent.as_tool()`, so parallelism is a property of the code, not of whether a Flash-Lite model
chooses to emit parallel tool calls. Every tool call is stamped with `started_at`/`ended_at`, and
the UI draws an agent timeline from them — overlapping bars are evidence, not decoration. Measured
speedups: **1.4x to 2.0x**.

**A second round exists to fill a stated gap, not to keep fishing.** Each researcher returns a
`gap` field, the Lead sees it beside the findings, and `MAX_FANOUT_ROUNDS = 2` is enforced in
Python — the tool refuses a third round rather than asking the model to stop. When a run opens a
second round the timeline says `2 rounds` and draws its researchers under the second `research`
call.

**Every budget that bounds a run is checked in Python**, not in a prompt: rounds, briefs per round,
turns per researcher, per-researcher timeout, and the total wall clock, all in `config.py`.
(Per-tool data limits — history length, news window, chart points — sit beside the tools that
enforce them, in `tools.py`.) `TOTAL_BUDGET_S = 240` with 45s reserved for the write-up, so no
question can run past four minutes regardless of what the models do — inside the five minutes a
grader waits. A model asked to respect its own budget sometimes will not, and the failure is
invisible.

## What It Does, and What It Cannot

Complete and deployed: the pair screen and holdout backtest, the reference database with its
Text2SQL surface, the Clarifier's fast path to a single Analyst, and the Clarifier → Lead →
Researchers → Writer pipeline with its citation audit.

Known limits:

- **A screen covers at most 40 names.** Screening is quadratic — 40 names is already 780 fits — and
  five GICS sectors are larger (Industrials 83, Financials 76, Information Technology 76, Health
  Care 60, Consumer Discretionary 47). Those screens take the first 40 alphabetically and return
  `universe_truncated` with the number dropped, so the agent reports a subset rather than the
  sector. To cover a large sector properly, screen its GICS sub-industries one at a time.
- Cboe rate-limits bursts, so a wide screen can report symbols it could not fetch.
- The deflated Sharpe haircut is large enough on a holdout of a few hundred sessions that few pairs
  could ever clear it — which the agent says explicitly rather than quietly dropping.

**What the honesty machinery looks like when it fires.** Three layers stop fabrication: a finding
must name the tool that produced it; the Writer has no tools, so it cannot reach data it was not
given; and a Python audit counts every number in the answer against its citation. Anything left
over is marked in place, never silently cut — `_[unsupported: more numbers stated than cited]_` in
a research answer, `[F1.2.1 - no such finding]` for an invented id, and a trailing "treat as
unverified" line on the fast path.

## Design Choices

- **Agent framework:** OpenAI Agents SDK (`openai-agents[litellm]`), routed to Gemini on Vertex AI.
  Tool schemas come from each function's signature and docstring.
- **Models:** the dropdown picks the model for every agent that does work — Lead, Researchers,
  Writer and the single Analyst. Only the Clarifier is pinned, to Gemini 3.5 Flash, because it runs
  on every question and a slow gate is worse than no gate. Flash-Lite is the default on
  measurement, not price: the same sector screen took 81s on Flash against 12–22s on Flash-Lite,
  and a single-pair question 38s against ~15s.
- **Sessions:** `SQLiteSession` over `conversations.db`, one row per message, so follow-ups work
  and sessions stay separate.
- **Separation of concerns:** `app.py` is FastAPI only; every agent decision lives behind
  `orchestrator.answer()`, the only seam between the web layer and the agents.
- **Analysis in Python:** every statistic — hedge ratio, spread, z-scores, half-life, rolling beta,
  the backtest and its deflated Sharpe — is computed in Python, not by the model.
- **Selection is in-sample too.** Freezing the backtest parameters is not enough if the *choice* of
  which pair to report was made using the holdout. Every screening statistic is measured on the
  in-sample window only, so the holdout is out of sample for the selection as well as the numbers.
- **Charts:** one library, Plotly, for every visualization: price lines, the screen's ranked table,
  and the spread's z-score panel with its entry and exit bands. Figures are built in Python as
  specs and rendered by the page, so they are interactive (zoom, pan, hover readout, drag to
  compare dates) and **never enter the model's context**, which would otherwise replay thousands of
  tokens on every later turn. Plotly is served from the installed package, not a CDN, so a blocked
  CDN cannot leave a grader with no charts.
- **UI:** a modern take on a Bloomberg terminal (black, amber, monospace), a startup price tape,
  and ten clickable sample queries grouped by what each exercises — hover one to see the agents and
  tools it will use.

## Sample Queries

Three to test with. Each names the source the answer comes from, what to expect, and the same
question put to the **same model with no tools and no system prompt**, so the difference is the
tools rather than what Gemini already knows.

Both columns were produced by `vertex_ai/gemini-3.5-flash-lite` on 2026-11-10. Quoted sentences are
verbatim; a long per-pair list is summarised, with its figures quoted exactly. Numbers move with
the market, so expect the *shape* of the answer to match rather than the digits.

---

### 1. "Which pairs in the energy sector are viable pair trades?"

**Source** — universe from `data/refdata.duckdb` (`instruments`, `gics_sector = 'Energy'`, 21
names); prices from Cboe,
`cdn.cboe.com/api/global/delayed_quotes/charts/historical/{symbol}.json`, 3 years of daily bars.

**Expect** — a screen of all 210 combinations, a ranked shortlist with half-life, correlation and
beta stability, a holdout backtest of the best few, and the conclusion that **none of them clears
its deflated Sharpe**. Roughly 25–35s, three researchers in parallel, charts, and an evidence list.

| With tools — Converge | Same model, no tools, no system prompt |
|---|---|
| (26s, 3 researchers in parallel, 1.8x speedup, 10 tool calls, 4 charts) "The evidence does not support an answer here regarding which pairs in the energy sector are viable, as all screened pairs demonstrated poor out-of-sample holdout performance or negative holdout Sharpe ratios." Then, per pair: MPC/VLO ranked highest with a rank score of **0.9583**, correlation **0.878**, half-life **20.4** sessions, holdout Sharpe **−0.59**, deflated Sharpe **−3.05**; COP/OXY holdout Sharpe **−0.51**, deflated **−2.78**, half-life **13.4** sessions, current z-score **7.48**; HAL/SLB, PSX/SLB, PSX/VLO and MPC/PSX likewise. Every number carries a `[F1.2.1]`-style id and an **Evidence** list naming the tool behind it. | (5,292 characters, **2 numeric tokens**) A confident sector essay: "Here are the most viable pairs in the energy sector", recommending **XOM/CVX** ("often called the ultimate energy pair"), SHEL/TTE, and others by sub-sector, with rationales like "divergence is usually driven by M&A announcements". No correlation, no half-life, no backtest, nothing measured. SHEL and TTE are NYSE-listed but are not S&P 500 constituents, so they are outside the reference universe and the agent could not have screened them even if asked. |

**The difference.** The bare model recommends pairs from memory of what pair trading *looks* like.
The agent fits all 210 combinations, finds the best ones on in-sample data only, tests them on
sessions the fit never saw, and reports that the evidence does not support a trade. The honest "no"
is the finding, and it is the answer the bare model cannot give.

---

### 2. "Is LMT a good pair trade with RTX?"

**Source** — Cboe daily bars for both legs,
`cdn.cboe.com/api/global/delayed_quotes/charts/historical/{LMT,RTX}.json`, 3 years.

**Expect** — a single named pair takes the **fast path**: one analyst, one `analyze_pair` call,
about 7s and a z-score chart, with no research fan-out. Expect a negative verdict with the hedge
ratio, half-life, beta stability and a holdout backtest.

| With tools — Converge | Same model, no tools, no system prompt |
|---|---|
| (7s, 1 tool call, 1 chart) "LMT and RTX do not make a strong pair trade, as evidenced by a losing out-of-sample backtest, weak correlation, and a slow mean-reversion half-life." Correlation **0.35**; half-life **61.3** days; beta stability **1.89**; holdout **−5.58%** total return, Sharpe **−0.34**, **33.3%** win rate over 3 trades in 226 sessions, exposure **38.9%**; current z-score **2.2**. Source and delay stated. | (4,204 characters) A well-informed fundamentals essay — RTX's Pratt & Whitney commercial exposure vs LMT's ~95% defence mix, the GTF powder-metal recall, F-35 programme risk — ending: "**Yes, LMT and RTX make a fundamentally sound pair.**" It even suggests the right method ("plot the historical price ratio... +/- 2 standard deviations") without being able to execute it. |

**The difference.** This is the sharpest case. The bare model's reasoning is genuinely good and its
conclusion is **the opposite of what the data says**. It says yes; the measured correlation is 0.35,
the half-life is 61 sessions, and the holdout loses money. Domain knowledge without measurement
produces a plausible wrong answer, which is worse than no answer.

---

### 3. "How many ETFs are in the reference database, and which ETF families are covered?"

**Source** — `data/refdata.duckdb`, table `instruments`, via the Text2SQL tools
(`get_reference_schema` then `query_reference`). The executed SQL is returned with the result so it
can be checked.

**Expect** — exactly **122** ETFs across **15** families, with counts: thematic 12, broad_us 12,
bond 12, international 12, vanguard_sector 11, sector_spdr 11, style 9, sp500_leveraged 8,
commodity 8, factor 7, nasdaq_leveraged 5, russell_leveraged 4, volatility 4, dow_leveraged 4,
equal_weight 3. About 7s, no fan-out.

| With tools — Converge | Same model, no tools, no system prompt |
|---|---|
| (7s, 2 tool calls) "There are **122** ETFs in the reference database (source: `instruments` reference table, delay: static)", then every family with its count, matching the table exactly. | (417 characters, **0 numeric tokens**) Cannot answer, and says so: "It looks like you are asking about a specific reference database, but you haven't mentioned *which* one!" |

*One honest note on that transcript: `delay: static` was never a real field. The prompt used to
tell every agent that results carry a `delay`, so the model supplied one for a table that has none.
That rule is now a presence test and the reference tools return `source` without `delay` — but the
recorded answer is left exactly as the run produced it, because a traceability demo that quietly
edits its own transcript is worth nothing.*

**The difference.** This is data the model cannot know — a database built by this project's own
offline step. The bare model correctly declines; the agent reads the schema, writes the SQL, and
returns the exact counts with the query attached.

---

### Also worth trying

Not extra sample queries — four inputs that each show a different graded behaviour. Every wording
below is one of the app's own clickable examples, so all four are known-good.

| Type this | What it shows |
|---|---|
| `Which pairs are viable?` | **Vague input:** the Clarifier asks one question back and stops. It can never ask twice in a conversation — the marker it prefixes to the reply is checked in Python, not in a prompt. |
| `Is there a good pair trade in bitcoin?` | **Out of scope,** said in the first sentence with what the desk can do instead. A separate verdict from vague, and it does not consume the one clarification. |
| `Compare information technology and financials: which sector has better pairs?` | Two sectors screened at once. Both hold 76 names against the 40-name cap, so watch the answer report a **subset** rather than claiming it covered the sector. |
| `Which energy pairs diverged the most, and what caused those divergences?` | The heaviest run: screen the sector, then news and SEC filings over exactly the divergence windows. Most likely to make the Lead open a **second round** on a stated gap. |

**A note on effort.** Queries 2 and 3 are answered in about 7 seconds by one analyst; query 1 fans
out to three researchers for about 26. The clarifier gate sizes the work to the question, so asking
for one chart does not pay for a research desk. The fan-out, its timeline and the `[F…]` ids appear
only on research-shaped questions — query 1 is the one to try.

## Data Sources

### The universe, and how to load or rebuild it

`data/refdata.duckdb` holds **625 instruments**: all **503** S&P 500 constituents plus **122**
hand-curated ETFs in 15 families (index trackers, sector SPDRs, Vanguard sectors, bonds,
commodities, factor and style funds, and the leveraged/inverse families with their multiples and
the fund each one tracks). 780 KB, committed, so it ships inside the container image and the
deployed app needs no setup.

Nothing needs loading — read it directly:

```bash
uv sync                                       # Python >= 3.10
duckdb data/refdata.duckdb "SELECT symbol, name, gics_sector FROM instruments LIMIT 5"
```

```python
import duckdb
rows = duckdb.connect("data/refdata.duckdb", read_only=True).execute(
    "SELECT symbol, name, gics_sector FROM instruments WHERE gics_sector = 'Energy'"
).fetchall()
```

Rebuild it only to refresh membership:

```bash
uv run build/build_refdata.py                 # under a minute
```

That reads S&P 500 membership, GICS sector, GICS sub-industry and company name from the Wikipedia
constituents table (parsed by `build/wikitables.py`, standard library only), takes each CIK from
the SEC's ticker file and falls back to Wikipedia's CIK column for recent listings, fills any
missing fund name from the Nasdaq Trader listing directory, and writes both the database and
`data/refdata_mappings.json` — the schema card, with per-column descriptions and real enum values,
that the Text2SQL tools hand the model, which is why `query_reference` can filter on sectors
without a prompt listing them. Prices are **not** stored: every correlation, hedge ratio, half-life
and backtest is computed live from Cboe, because those change daily and a stale copy would be worse
than none.

The curated ETF families, their leverage multiples and the fund each one tracks exist nowhere but
this build, which is why the no-tools column in query 3 cannot answer at all.

An earlier version read sectors from the SEC's SIC codes instead. SEC rate-limits per IP hard
enough that a patient four-thread crawl resolved **12 of 503**, while the Wikipedia table carries
GICS for all 503 in a single page fetch — and GICS is the vocabulary traders actually use.

### The feeds

| Source | What it provides | Freshness |
|---|---|---|
| Cboe delayed quotes | Daily OHLCV for US stocks, ETFs and indices, 2004–present | ~15 min delayed; the last close when the market is shut |
| SEC EDGAR | 8-K events with item codes, 10-K and 10-Q filings, to explain a divergence | Same day |
| Google News RSS | Headlines for a symbol or keywords; explains prolonged divergences | Same day, headlines only |
| Nasdaq Trader | Symbol directory: listings, names and ETF flags | Daily |
| Wikipedia | S&P 500 constituents with GICS sector and sub-industry (offline build only) | Monthly, in practice |

**Not covered:** options, crypto, futures, Form 4 executive transactions, fundamentals, rates and
FX. Prices are split-adjusted but exclude dividends, so every return is a price return — which
flatters a short leg and penalises a long one.

## Tools

| Tool | Purpose | Source |
|---|---|---|
| `screen_pairs` | Fit every combination of a resolved universe (sector, ETF family or symbol list, capped at 40 names), rank them, and backtest the best on a holdout | Cboe + reference DB |
| `analyze_pair` | One pair in full: statistics, holdout backtest, divergence episodes and a z-score chart | Cboe |
| `research_divergence` | News and SEC filings for both legs over exactly one divergence window | Google News, SEC EDGAR |
| `get_reference_schema` | Columns and real enum values of the reference table, for writing SQL | Reference DB |
| `query_reference` | One read-only SELECT against the reference table; returns the SQL it ran, for citation | Reference DB |
| `get_price_history` | Period return, high, low, realized volatility, the last five closes, and a month-end price table | Cboe |
| `get_chart` | Interactive daily closing-price line chart, on request | Cboe |
| `get_news` | Headlines for a symbol or keywords: last 7 days, or a date range up to 14 days | Google News |
| `get_filings` | SEC 8-K events and 10-K/10-Q reports: latest, or a date range | SEC EDGAR |

## Run Locally

Requires a Google Cloud project with Vertex AI enabled.

```bash
gcloud auth application-default login
uv run app.py
```

Then open http://localhost:8000. If it cannot resolve the project, set
`GOOGLE_CLOUD_PROJECT=<your project id>`. No API keys and no other environment variables.

## Cloud Deployment

Deployed on Google Cloud Run behind Identity-Aware Proxy, built from `main` by a Cloud Build
trigger.

**https://tool-calling-agent-git-llmb6vmaoa-nn.a.run.app** — sign in with your Columbia Google
account when prompted. If access is refused, email jay2147@columbia.edu and it will be granted; the
app also runs locally with the two commands above.

There is no Dockerfile or `cloudbuild.yaml` in this repo: the trigger builds with Google Cloud
buildpacks and supplies the entrypoint as a substitution. The settings below live in the service
rather than in version control, so they are recorded here.

```
_ENTRYPOINT = uvicorn app:app --host 0.0.0.0 --port $PORT
_SERVICE_NAME = tool-calling-agent-git
```

| Setting | Value | Reason |
|---|---|---|
| `--timeout` | 900s | The default is 300s, exactly the grading window, so a slow screen would race the proxy |
| `--memory` | 1Gi | Measured 307 MB at idle from imports alone, and ~400 MB during a screen. The 512 MB default leaves no room |
| `--cpu` | 2 | Fitting every pair is CPU-bound: the 40-name cap is 780 regressions plus rolling betas |
| `--min-instances` | 1 | Keeps one warm instance, so chat history survives between requests. Bills continuously; set back to 0 when not being marked |
| `--max-instances` | 1 | Sessions live in one instance's memory, so a second instance would not see them |

**State on Cloud Run.** The reference database ships inside the image and is opened read-only — no
setup, no writes, no WAL. `conversations.db` does not: Cloud Run's filesystem is in memory, so
conversation history is per-instance and is lost on a restart, a new revision or a scale to zero.
That is acceptable because the agent is stateless between questions apart from the transcript, but
it is a real limitation rather than a hidden one.

## Files

```
app.py        FastAPI only: the page, /chat, /clear, the price tape, the Plotly bundle.
              Do not rename this file or the `app` object — the deploy entrypoint is `uvicorn app:app`.
config.py     Which model each role uses, and the caps on one run
tools.py      The nine tools the model can call: plain Python bodies, @function_tool wrappers
chart.py      Themed Plotly specs: lines, tables, and the z-score panel
index.html    The UI

pipeline/     orchestrator (the seam), roster (the five agents and every instruction they are
              given), research (the asyncio.gather fan-out), citations (the Python audit),
              schemas (the structured hand-offs), runtime (per-run state the model never sees)
quant/        pairs (hedge ratio, spread, z-scores, half-life, rolling beta), backtest (holdout,
              frozen parameters, deflated Sharpe), screen (every combination, ranked)
sources/      data (Cboe, SEC EDGAR, Google News, Nasdaq Trader), refdata (read-only DuckDB)

data/         The committed reference database and its schema card
build/        Offline, run once. build_refdata.py writes data/; wikitables.py parses the S&P 500
              constituents table. Never imported at request time
```

## Future Improvements

Ordered roughly by how much they would improve the experience rather than by effort.

**Showing the work while it happens**

- **Live progress during the wait.** A research-shaped question takes 25–35s and the page shows
  only an elapsed counter; the run's structure is revealed afterwards.
- **SSE streaming.** A `GET /stream` side channel so each step reaches the page as it happens, with
  the run owned by `POST /chat` so closing the tab cannot kill it.
- **Live agent graph.** Promote the post-hoc timeline into a graph that draws itself during the run.
- **Researcher columns.** Each researcher's brief, tools and findings side by side while they work.

**Agent architecture**

- **A Critic, and an iterative review cycle.** An agent whose job is to attack a result that looks
  too good — challenge a high Sharpe, a short holdout, a suspiciously clean equity curve — and send
  the Lead back for another round. It fits the overfitting theme exactly, and it is the natural next
  agent shape after Lead/Researcher/Writer.
- **Multi-turn clarification.** The Clarifier asks at most one question, by design, so it can never
  loop. A richer version would ask a short series — universe, then window, then what "good" means.
- **`[F…]` ids on the fast path.** The analyst path already prints an **Evidence** list of the calls
  behind its answer and flags any number absent from this run's tool results, but it has no findings
  store, so it carries no per-number ids. Both paths should cite identically.

**Evaluation**

- **An eval suite.** None exists today. The valuable set is not accuracy on a known answer but
  discipline: that a vague question is clarified exactly once, that no number appears without a
  citation, that a fabricated finding is caught, and that a random-walk pair is never called
  tradeable. Then ablations — no fan-out, no gap loop, no citation audit — to show the architecture
  earns its complexity.

**Data and analysis**

- **More granular pricing.** Intraday history would allow entry and exit timing within the session,
  realistic slippage, and a check on whether a divergence opened on one print or over hours.
- **Basket vs basket.** Cointegrated baskets on both sides, including sparse replication of a
  sector ETF by a handful of its constituents.
- **Futures legs.** Basis trades such as VIX futures against VXX, or ES against SPY.
- **Dividend-adjusted returns.** Every figure is a price return today.
- **Persistent sessions.** Several conversations at once, surviving a restart, which needs storage
  outside the container's in-memory filesystem.
