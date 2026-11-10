# Converge: A Pair-Trading Research Agent

IEOR-E4570: Project 2 — Deep Research

Author: Joseph Young (jay2147)

## The Question

> Given a sector, an ETF family, or a user's own idea, which pairs of US stocks and ETFs are viable
> pair trades — correlated names whose prices move together, drift apart, then converge again? And
> when a pair stays diverged for an unusually long stretch, what news explains it: a temporary
> dislocation, or a relationship that has actually broken?

Converge screens every combination up to a set limit, backtests the strongest candidates on data
they were not chosen on, and ranks them on several metrics. Pairs are selected by Python, never by
the model: the agent decides *what to investigate*, and the arithmetic decides *what survives*.

For research only, not investment advice. Always double-check results.

## How It Works

A question goes through four kinds of agent. The division is the point: each one is denied
something, and what it is denied is what stops it inventing numbers.

```
question
   |
   v
Clarifier   no tools, one turn, structured verdict
   |          vague -> ask one question back and stop
   |          out of scope (crypto, options, FX) -> say so and stop
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
`Agent.as_tool()`, because "several researchers in one turn" is the graded requirement and should
be a property of the code rather than of whether a Flash-Lite model chooses to emit parallel tool
calls. Every tool call is stamped with `started_at`/`ended_at`, and the UI draws an agent timeline
from them — overlapping bars are evidence, not decoration. Measured speedups: **1.4x to 2.0x**.

**Every cap lives in `config.py`, not in a prompt.** Research rounds, briefs per round, turns per
researcher, per-researcher timeout and the total wall-clock budget are all checked in Python. A
model asked to respect its own budget sometimes will not, and the failure is invisible.

## Status

Complete and deployed: the pair screen and holdout backtest, the reference database with its
Text2SQL surface, and the Clarifier → Lead → Researchers → Writer pipeline with its citation audit.

Known limits, stated rather than hidden: conversation history is per-instance (Cloud Run's
filesystem is in memory), price returns exclude dividends, Cboe rate-limits bursts so a wide screen
can report symbols it could not fetch, and the deflated Sharpe haircut is large enough on a
three-year holdout that few pairs could ever clear it — which the agent says explicitly rather than
quietly dropping.

## Design Choices

- **Agent framework:** OpenAI Agents SDK (`openai-agents[litellm]`), routed to Gemini on Vertex AI.
  Tool schemas come from each function's signature and docstring, replacing Project 1's hand-written
  schema list.
- **Models:** reasoning and prose are fixed on Gemini 3.5 Flash. The researcher model is selectable
  in the UI from four Gemini models, because that is where the cost/quality trade-off lives.
- **Sessions:** `SQLiteSession` over `conversations.db`, one row per message, so follow-ups work and
  sessions stay separate. On Cloud Run the filesystem is in memory, so history lasts as long as the
  instance.
- **Separation of concerns:** `app.py` is FastAPI only; every agent decision lives behind
  `orchestrator.answer()`, which is the single seam the multi-agent pipeline grows into.
- **Analysis in Python:** every statistic — hedge ratio, spread, z-scores, half-life, rolling beta,
  the backtest and its deflated Sharpe — is computed in Python, not by the model. The model chooses
  *what to investigate*; arithmetic decides *what survives*.
- **Selection is in-sample too.** Freezing the backtest parameters is not enough if the *choice* of
  which pair to report was made using the holdout. Every screening statistic is measured on the
  in-sample window only, so the holdout is out of sample for the selection as well as the numbers.
- **Three layers against fabrication:** a finding must name the tool that produced it; the writer
  has no tools, so it cannot reach data it was not given; and a Python audit counts every number in
  the answer against its citation. Anything left over is marked in the text, never silently cut.
- **Charts:** one library, Plotly, for every visualization — lines and tables today, spread and
  correlation views next. Figures are built in Python as specs and rendered by the page, so they are
  interactive (zoom, pan, hover readout, drag to compare dates) and **never enter the model's
  context**, which would otherwise replay thousands of tokens on every later turn. Plotly is served
  from the installed package, not a CDN, so a blocked CDN cannot leave a grader with no charts.
- **UI:** a modern take on a Bloomberg terminal (black, amber, monospace) with a live price tape and
  clickable sample queries.

## Who It's For

Anyone weighing a statistical pair trade who wants the evidence and its limits rather than a
recommendation: analysts, quantitative finance students, and traders checking whether a pair
survives costs, a holdout and a multiple-testing correction. The agent is built to say "the
evidence does not support an answer here" and mean it.

## Sample Queries

Three to test with. Each names the source the answer comes from, what to expect, and — as the
assignment asks — the same question put to the **same model with no tools and no system prompt**,
so the difference is attributable to the tools rather than to what Gemini already knows.

Both columns were produced by `vertex_ai/gemini-3.5-flash-lite`. The agent's answers below are
verbatim from live runs on 2026-11-10; numbers move with the market, so expect the *shape* of the
answer to match rather than the digits.

---

### 1. "Which pairs in the energy sector are viable pair trades?"

**Source** — universe from `data/refdata.duckdb` (`instruments`, `gics_sector = 'Energy'`, 21
names); prices from Cboe,
`cdn.cboe.com/api/global/delayed_quotes/charts/historical/{symbol}.json`, 3 years of daily bars.

**Expect** — a screen of all 210 combinations, a ranked shortlist with half-life, correlation and
beta stability, a holdout backtest of the best few, and the conclusion that **none of them clears
its deflated Sharpe**. Roughly 25–35s, three researchers in parallel, charts, and an evidence list.

| | |
|---|---|
| **Converge** (26s, 3 researchers in parallel, 1.8x speedup, 10 steps, 4 charts) | "The evidence does not support an answer here regarding which pairs in the energy sector are viable, as all screened pairs demonstrated poor out-of-sample holdout performance or negative holdout Sharpe ratios." Then, per pair: MPC/VLO ranked highest with a rank score of **0.9583**, correlation **0.878**, half-life **20.4** sessions, holdout Sharpe **−0.59**, deflated Sharpe **−3.05**; COP/OXY holdout Sharpe **−0.51**, deflated **−2.78**, half-life **13.4** sessions, current z-score **7.48**; HAL/SLB, PSX/SLB, PSX/VLO and MPC/PSX likewise. Every number carries a `[F1.2.1]`-style id and an **Evidence** list naming the tool behind it. |
| **Same model, no tools** (5,292 characters, **2 numeric tokens**) | A confident sector essay: "Here are the most viable pairs in the energy sector", recommending **XOM/CVX** ("often called the ultimate energy pair"), SHEL/TTE, and others by sub-sector, with rationales like "divergence is usually driven by M&A announcements". No correlation, no half-life, no backtest, nothing measured — and two of its named pairs are not US-listed, so they are outside any universe it could have screened. |

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

| | |
|---|---|
| **Converge** (7s, 1 tool call, 1 chart) | "LMT and RTX do not make a strong pair trade, as evidenced by a losing out-of-sample backtest, weak correlation, and a slow mean-reversion half-life." Correlation **0.35**; half-life **61.3** days; beta stability **1.89**; holdout **−5.58%** total return, Sharpe **−0.34**, **33.3%** win rate over 3 trades in 226 sessions, exposure **38.9%**; current z-score **2.2**. Source and delay stated. |
| **Same model, no tools** (4,204 characters) | A well-informed fundamentals essay — RTX's Pratt & Whitney commercial exposure vs LMT's ~95% defence mix, the GTF powder-metal recall, F-35 programme risk — ending: "**Yes, LMT and RTX make a fundamentally sound pair.**" It even suggests the right method ("plot the historical price ratio... +/- 2 standard deviations") without being able to execute it. |

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

| | |
|---|---|
| **Converge** (7s, 2 tool calls) | "There are **122** ETFs in the reference database (source: `instruments` reference table, delay: static)", then every family with its count, matching the table exactly. |
| **Same model, no tools** (417 characters, **0 numeric tokens**) | Cannot answer, and says so: "It looks like you are asking about a specific reference database, but you haven't mentioned *which* one!" |

**The difference.** This is data the model cannot know — a database built by this project's own
offline step. The bare model correctly declines; the agent reads the schema, writes the SQL, and
returns the exact counts with the query attached.

---

**A note on effort.** Queries 2 and 3 are answered in about 7 seconds by one analyst, while query 1
fans out to three researchers for about 26. That is deliberate: a clarifier gate sizes the work to
the question, so asking for one chart does not pay for a research desk. The fan-out, its timeline
and the `[F…]` citation system appear on research-shaped questions — query 1 is the one to try if
you want to see them.

## Data Sources

### The universe, and how to rebuild it

`data/refdata.duckdb` holds **625 instruments**: all **503** S&P 500 constituents plus **122**
hand-curated ETFs in 15 families (index trackers, sector SPDRs, Vanguard sectors, bonds,
commodities, factor and style funds, and the leveraged/inverse families with their multiples and
the fund each one tracks). 780 KB, committed, so it ships inside the container image and the
deployed app needs no setup.

It is built offline, never at request time:

```bash
uv run build/build_refdata.py          # ~10 seconds
```

That reads S&P 500 membership, GICS sector, GICS sub-industry, company name and CIK from the
Wikipedia constituents table (parsed by `build/wikitables.py`, standard library only), fills any
missing fund name from the Nasdaq Trader listing directory, and writes both the database and
`data/refdata_mappings.json` — the schema card, with real enum values, that the Text2SQL tools give
the model. Prices are **not** stored: every correlation, hedge ratio, half-life and backtest is
computed live from Cboe, because those change daily and a stale copy would be worse than none.

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
| Wikipedia | S&P 500 constituents with GICS sector, sub-industry and CIK (offline build only) | Monthly, in practice |

All are free public feeds with no API keys.

**Not covered:** options, crypto, futures, Form 4 executive transactions, fundamentals, rates and
FX. Prices are split-adjusted but exclude dividends, so every return is a price return.

## Tools

| Tool | Purpose | Source |
|---|---|---|
| `screen_pairs` | Fit **every** combination in a sector, ETF family or symbol list, rank them, and backtest the best on a holdout | Cboe + reference DB |
| `analyze_pair` | One pair in full: statistics, holdout backtest, divergence episodes and a z-score chart | Cboe |
| `research_divergence` | News and SEC filings for both legs over exactly one divergence window | Google News, SEC EDGAR |
| `get_reference_schema` | Columns and real enum values of the reference table, for writing SQL | Reference DB |
| `query_reference` | One read-only SELECT against the reference table; returns the SQL it ran, for citation | Reference DB |
| `get_price_history` | Period return, high, low and realized volatility over a date range | Cboe |
| `get_chart` | Interactive daily closing-price line chart, on request | Cboe |
| `get_news` | Headlines for a symbol or keywords: last 7 days, or a date range | Google News |
| `get_filings` | SEC 8-K events and 10-K/10-Q reports: latest, or a date range | SEC EDGAR |

## Run Locally

Requires a Google Cloud project with Vertex AI enabled.

```bash
gcloud auth application-default login
uv run app.py
```

Then open http://localhost:8000. If it cannot resolve the project, set
`GOOGLE_CLOUD_PROJECT=<your project id>`.

No API keys and no other environment variables. Every feed is called with one generic
User-Agent that carries no personal contact details.

## Cloud Deployment

Deployed on Google Cloud Run, built from `main` by a Cloud Build trigger. Sign in with a Columbia
Google account.

- https://tool-calling-agent-git-llmb6vmaoa-nn.a.run.app
- Backup: https://tool-calling-agent-git-401765433529.northamerica-northeast1.run.app

Service settings that matter, and why:

| Setting | Value | Reason |
|---|---|---|
| `--timeout` | 900s | The default is 300s, exactly the grading window, so a slow screen would race the proxy |
| `--memory` | 1Gi | Measured 307 MB at idle from imports alone, and ~400 MB during a screen. The 512 MB default leaves no room |
| `--cpu` | 2 | Fitting every pair is CPU-bound: a 40-symbol universe is 780 regressions plus rolling betas |
| `--min-instances` | 1 | Keeps one warm instance, so chat history survives between requests. Bills continuously; set back to 0 when not being marked |
| `--max-instances` | 1 | Sessions live in one instance's memory, so a second instance would not see them |

**Two things about state on Cloud Run.** The reference database (`data/refdata.duckdb`) is committed,
so it ships inside the container image and is opened read-only - no setup, no writes, no WAL.
Conversation history (`conversations.db`, SQLite) is different: Cloud Run's filesystem is in-memory,
so history is per-instance and is lost on a restart, a new revision or a scale to zero. That is
acceptable here because the agent is stateless between questions apart from the transcript, but it
is a real limitation rather than a hidden one.

## Files

| File | Role |
|---|---|
| `app.py` | FastAPI only: the page, `/chat`, `/clear`, the price tape, the Plotly bundle |
| `orchestrator.py` | `answer()`: one question in, one answer out. The seam the agent pipeline grows into |
| `roster.py` | The agents, their instructions and the Vertex model factory |
| `config.py` | Which model each role uses, and the caps on one run |
| `tools.py` | The tools the model can call: plain Python bodies, with `@function_tool` wrappers |
| `pairs.py` | The pair maths: hedge ratio, spread, z-scores, half-life, rolling beta, divergence episodes |
| `backtest.py` | Holdout backtest: fit in-sample, trade both windows on frozen parameters, deflate the Sharpe |
| `screen.py` | Fit every combination in a universe, rank them, backtest the best |
| `refdata.py` | Read-only DuckDB over the reference table: schema, enum values, SQL, universe resolution |
| `hooks.py` | Records every tool call, tagged with the agent that made it |
| `run_context.py` | Per-run state the model never sees, such as figures bound for the page |
| `chart.py` | Themed Plotly figure specs (lines, tables and the z-score panel) |
| `data.py` | Fetchers for Cboe, SEC EDGAR, Google News and Nasdaq Trader |
| `index.html` | The UI |
| `data/` | The committed reference database and its schema card |
| `build/` | **Offline, run once.** `build_refdata.py` writes `data/`; `wikitables.py` parses the S&P 500 constituents table. Never imported at request time |

## Future Improvements

- **Futures legs:** basis trades (VIX futures vs VXX, ES vs SPY) need a keyed data vendor
- **Basket vs basket:** cointegrated baskets on both sides, not just a single pair
- **Intraday execution:** entry and exit timing within the session, which needs tick history
- **Richer evaluation:** a larger eval suite and ablations over the research architecture
- **Persistent sessions:** several chats open at once, surviving a restart
