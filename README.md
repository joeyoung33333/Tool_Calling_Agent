# Converge: A Pair-Trading Research Agent

IEOR E4570: Project 2 — Deep Research

Author: Joseph Young (jay2147)

**Live:** https://tool-calling-agent-git-llmb6vmaoa-nn.a.run.app (access details under
[Cloud Deployment](#cloud-deployment))

$\color{red}\textsf{\textbf{Late Changes due to Cboe blocking this app on 2026-11-19.}}$<br>
$\color{red}\textsf{\textbf{The app was deployed and working on time, before the block.}}$<br>
$\color{red}\textsf{\textbf{Update to switch out underlying pricing source to Nasdaq, with Yahoo Finance as a fallback.}}$

**Error:** every price request failed with `cdn.cboe.com has no data for this request: the symbol is
not available on this feed.` **Reason:** Cboe moved its data behind Cloudflare, which redirects
`cdn.cboe.com` to `cdn-api.cboe.com` and answers this app with HTTP 403, "Sorry, you have been
blocked".

The three test questions with expected answers: [Sample Queries](#sample-queries).

## The Question

> Given a sector, an ETF family, or an idea of your own, which pairs of US stocks and ETFs are
> viable pair trades — names that track each other, drift apart, then converge again? And when a
> pair stays apart for an unusually long stretch, does the news explain a temporary dislocation or
> a relationship that has actually broken?

Converge draws up to 40 names from its reference database (the S&P 500 plus 122 curated ETFs),
fetches three years of daily prices for them from Nasdaq, fits and ranks every pair on the first 70%
of that history (about two years), then backtests the best pairs on the final 30% (about eleven
months). When a pair stays diverged, it pulls SEC filings and news headlines from that window to
explain why. The agents decide what to investigate and which tools to call with which arguments;
Python runs the backtests and computes the statistics.

Research only, not investment advice.

## Who It's For

For analysts, finance students, and traders who want pair-trading ideas in US stocks and ETFs
researched and tested rather than recommended: whether a pair survives trading costs, a holdout, and
a multiple-testing correction. Converge starts from the adversarial assumption that a pair is not
viable until the holdout says otherwise, and it reports "the evidence does not support an answer
here" when that is the finding.

## How It Works

Five agents across two paths. The **Clarifier** agent reads every question first: it asks back once
if the question is vague, declines if it is out of scope, and otherwise sends it down one of two
paths:

- **Fast path** (~10–40s): one **Analyst** agent handles small requests, such as a single pair or a
  chart, with all nine data tools.
- **Research path** (~2 minutes): a **Lead** splits the question into briefs; up to four
  **Researchers** screen the reference database and pull daily prices, news and SEC filings in
  parallel; and a **Writer** turns the findings into the answer.

<pre>
question
   |
   v
<b><span style="color:#ff8a3d">Clarifier</span></b>  [Runner.run]       <i>no tools, one turn: asks back if vague, declines if out of scope</i>
   |
   +-- fast path --&gt; <b><span style="color:#c08bff">Analyst</span></b>  [Runner.run]   <i>9 data tools; numbers checked against tool results</i>
   |
   v  research path
<b><span style="color:#ffb000">Lead</span></b>       [Runner.run]       <i>research() and write_report() only; no data tools</i>
   |
   +-- research()  [custom tool call] --&gt; asyncio.gather, one Runner.run per Researcher
   |                                      |-- <b><span style="color:#c08bff">Researcher 1</span></b>  [Runner.run]
   |                                      |-- <b><span style="color:#9a6bdd">Researcher 2</span></b>  [Runner.run]
   |                                      |-- <b><span style="color:#d9b3ff">Researcher 3</span></b>  [Runner.run]
   |                                      \-- <b><span style="color:#7b4fc0">Researcher 4</span></b>  [Runner.run]
   |                                      <i>in parallel; each: 9 data tools, own context, one brief</i>
   |
   |&lt;-- <i>findings, with citation ids minted in Python</i>
   |
   +-- <i>at most one more round, to fill a stated gap</i>
   |
   +-- write_report()  [custom tool call]  <i>records the question and ends the Lead's run</i>
   |
   v
<b><span style="color:#5ab0ff">Writer</span></b>     [Runner.run]       <i>no tools; sees only the question and the findings</i>
   |
   v
citation audit                <i>every number counted against its citation</i>
   |
   v
answer + evidence list

Key   <b>bold</b>    an agent:  <b><span style="color:#ff8a3d">Clarifier</span></b>  <b><span style="color:#c08bff">Analyst</span></b>  <b><span style="color:#ffb000">Lead</span></b>  <b>Researchers</b> <b><span style="color:#c08bff">1</span></b> <b><span style="color:#9a6bdd">2</span></b> <b><span style="color:#d9b3ff">3</span></b> <b><span style="color:#7b4fc0">4</span></b>  <b><span style="color:#5ab0ff">Writer</span></b>
      plain   a Python step: routing, asyncio.gather, the citation audit
      [ ]     how it runs: Runner.run starts an agent; a custom tool call is a function we wrote
              for the Lead's model to call; no handoffs
      <i>italic</i>  what that step does, or may use
</pre>

Built on the OpenAI Agents SDK, routed to Gemini on Vertex AI. The Lead fans out in one turn: a
single `research()` call runs every brief at once with `asyncio.gather`, and each Researcher is its
own `Runner.run` with a fresh context. That is not an SDK handoff, which would hand it the whole
conversation, or `Agent.as_tool()`, which would leave the parallelism up to the model. Every call is
stamped with its start and end time, which the page draws as a timeline. The Lead gets at most one
more round, aimed at a `gap` the Researchers reported (`MAX_FANOUT_ROUNDS = 2`). The Writer runs on
60 seconds held back for it, and the caps on rounds, turns and time live in `config.py`, so a run
finishes inside the five-minute grading window.

Citations are checked in Python. Each finding names the tool that produced it and gets an id such as
`[F1.2.1]` minted in Python, and the audit checks every number against the finding it cites.
Anything that fails is marked in place: `[unsupported: more numbers stated than cited]`,
`[unsupported: does not match the finding it cites]`, or `[F1.2.1 - no such finding]` for an
invented id. On the fast path, any number found in no tool result is listed as unverified above the
evidence list.

| Agent | Model | Reasoning | Why |
|---|---|---|---|
| Clarifier | Gemini 3.5 Flash, pinned; retried on 3.5 Flash-Lite after 12s | Medium | Decides the route |
| Lead | Gemini 3.5 Flash, pinned; retried on 3.5 Flash-Lite if it fails before research | Medium | Plans the research |
| Writer | Gemini 3.5 Flash, pinned; retried on 3.5 Flash-Lite if it fails | Medium | Writes the answer |
| Analyst | Dropdown | Medium | Makes most of the calls |
| Researchers | Dropdown | Medium | Make most of the calls |

*Dropdown options: Gemini 3.5 Flash-Lite (default), 3.8 Flash, 3.5 Flash and 3.1 Flash-Lite.*

## The Interface

A Bloomberg-terminal take on a research desk: black, amber, monospace, with a scrolling tape of last
closes.

- **Ten clickable examples** in three groups, one per path. Hover one to see which agents
  and tools it will use.
- **Stop and start over.** While a run is in flight, SEND becomes **STOP**, which abandons it in the
  page. **New session** also clears the conversation on the server.
- **Model dropdown**, switchable mid-conversation; follow-up questions keep their context.
- **Run timeline.** An elapsed-time counter while it works, then one color per agent on a shared
  clock. Rows expand to show the tool calls, their arguments and results, and any SQL. A failed
  step is drawn in red, followed by its retry in the agent's color.
- **Citations you can check.** Hover an `[F1.2.1]` to see the finding behind it; the full evidence
  list sits collapsed under the answer.
- **Charts** in tabs when there are several, each captioned with how to read it. Hover for a value;
  drag to zoom, and the note underneath reports the move over the visible window (in z for a
  z-score panel). The screen table labels its fit and holdout columns and says what each one is.
- **Other:** Up and Down recall earlier prompts; a tape symbol shows its day and 52-week range on
  hover; the tape and twinkle background have speed controls.

## Limitations

- **A screen covers at most 40 names.** Five GICS sectors are larger (Industrials holds 83), so
  their screens take the first 40 by ticker and report the number dropped; to cover one, screen
  its sub-industries one at a time.
- **Unofficial price feeds.** Nasdaq and Yahoo Finance publish no limits for these endpoints, so a
  wide screen can report symbols it could not fetch. Indices are not covered; use their ETFs.
- **A strict bar.** A pair is viable only if its holdout Sharpe beats what the luckiest of the pairs
  backtested would post by chance: about 2.2 for the default eight on an 11-month holdout. Few pairs
  clear it, and the agent says so rather than dropping them.
- **Price returns only.** Prices are split-adjusted but exclude dividends, which flatters a short
  leg and penalizes a long one.
- **Model latency.** A Gemini call occasionally stalls for a minute. Each call gives up after 25s
  (the Clarifier after 12s, the Writer after 45s), and the pinned agents retry on Flash-Lite. If time
  still runs out, the run says so and answers from what it had gathered.
- **Not covered:** options, crypto, futures, FX, rates, indices, fundamentals, Form 4 insider trades
  and intraday bars. Prices are end of day: the last bar is the last completed session.

## Sample Queries

The UI's three research-path examples, each fanned out to parallel researchers, beside the same
question to the Writer's model (Gemini 3.5 Flash) with **no tools and no system prompt**. Recorded
2026-11-20, quoted verbatim; numbers move with the market, so expect the same shape, not digits.

---

### 1. "Which pairs in the energy sector are viable pair trades?"

**Path** — Clarifier → Lead → 3–4 Researchers in parallel → Writer → citation audit

**Tool calls** — `screen_pairs` once per brief, on the whole sector or one sub-industry, after
`get_reference_schema` and `query_reference` resolve the sub-industries; `analyze_pair` and
`research_divergence` for a named pair. The recorded run made 14 calls.

**Source** — universe from `data/refdata.duckdb`, table `instruments` (GICS sector Energy, 21
names); end-of-day market data from Nasdaq,
`api.nasdaq.com/api/quote/{symbol}/historical` (Yahoo Finance if Nasdaq fails), 3 years of daily bars.

**Expect** — **no pair with a positive deflated Sharpe**, each reported with its holdout Sharpe and
deflated Sharpe. About two minutes.

| Converge (114s) | Gemini 3.5 Flash, no tools |
|---|---|
| "No pairs in the screened Energy sector are viable pair trades as none achieved a positive deflated Sharpe ratio." E.g. COP/OXY: holdout Sharpe **−1.19**, deflated Sharpe **−3.46**. | "Here are the most viable pairs in the energy sector", led by **ExxonMobil vs. Chevron**, all from memory. |

---

### 2. "Are these viable pair trades: AMZN vs GOOGL, MS vs JPM, WMT vs TGT, KO vs PEP, and SPY vs QQQ?"

**Path** — Clarifier → Lead → 3–4 Researchers in parallel (one or two pairs each) → Writer →
citation audit

**Tool calls** — `analyze_pair` once per pair: 5 calls.

**Source** — end-of-day market data from Nasdaq for the ten tickers,
`api.nasdaq.com/api/quote/{symbol}/historical` (Yahoo Finance if Nasdaq fails), 3 years of daily bars.

**Expect** — **nothing confirmed viable**: AMZN/GOOGL positive but undeflated, MS/JPM and SPY/QQQ
losing on the holdout, WMT/TGT and KO/PEP rejected for a negative hedge ratio. About 1–2 minutes.

| Converge (90s) | Gemini 3.5 Flash, no tools |
|---|---|
| "None of the evaluated pairs are viable pair trades as none of them have a positive deflated Sharpe ratio." AMZN/GOOGL holdout Sharpe **2.21**, uncorrected; MS/JPM **−1.88**; SPY/QQQ **−1.52**; WMT/TGT and KO/PEP hedge ratios **−0.5356** and **−0.1632**, so not tradeable. | "Yes, all five of these are viable pairs trades", with KO vs PEP rated "Excellent (The Gold Standard)". |

---

### 3. "Which energy pairs diverged the most, and what caused those divergences?"

**Path** — Clarifier → Lead → 4 Researchers → Lead → 4 more Researchers for the gap → Writer →
citation audit

**Tool calls** — `screen_pairs` to find the pairs, `analyze_pair` for each pair's divergence
episodes, `research_divergence` over each episode's window. The recorded run made 29 calls.

**Source** — end-of-day market data from Nasdaq and the universe from `data/refdata.duckdb`, as in
query 1, plus Google News RSS (`news.google.com/rss/search`) and SEC EDGAR
(`data.sec.gov/submissions/CIK{cik}.json`) over each divergence window.

**Expect** — the widest divergence named from the spread z-scores, and a plain statement when the
news and filings do not settle the cause. About two minutes.

| Converge (143s) | Gemini 3.5 Flash, no tools |
|---|---|
| "…ConocoPhillips and Occidental Petroleum (COP/OXY) diverged the most with a peak z-score of **9.87**, though the findings do not contain news or SEC filings to analyze the causes of this divergence." For XOM/CVX it cites a January 2026 headline on Exxon's record high. | Reads "energy pairs" as commodities and recounts history: Henry Hub vs TTF gas, Brent vs WTI, crude vs natural gas, the crack spread. |

---

### Also Worth Trying

The fast-path and clarification examples from the UI, each recorded on 2026-11-20.

| Question | Tools and data | Expect |
|---|---|---|
| `Is AMZN a good pair trade with NVDA?` | `analyze_pair`; Nasdaq bars | Fast path, ~20–40s: correlation, half-life, a holdout backtest and a z-score chart. A positive holdout Sharpe (**2.15** over 5 trades): promising at most, never viable, because one pair is not deflated. |
| `How many ETFs are in the reference database, and which ETF families are covered?` | `get_reference_schema`, `query_reference`; `data/refdata.duckdb` (`instruments`) | ~15–25s: **122** ETFs in **15** families, each family's count from the agent's own GROUP BY. The SQL shows in the timeline. |
| `Chart AAPL's price over the last 6 months.` | `get_chart`; Nasdaq bars | ~20s: an interactive closing-price chart and the period return. |
| `How volatile has NVDA been over the past year?` | `get_price_history`; Nasdaq bars | ~20s: realized volatility, period return, high and low, and a month-end price table. |
| `What news and SEC filings came out for TSLA this month?` | `get_news`, `get_filings`; Google News, SEC EDGAR | ~20–30s: the month's headlines, and its 8-K, 10-K and 10-Q filings or a note that there were none. |
| `Find a good cryptocurrency pair trade?` | None; Clarifier only | ~5–15s: declined as out of scope in the first sentence, with what the desk can do instead. |
| `Chart this symbol.`, then `NVDA` | None, then `get_chart`; Nasdaq bars | ~5–15s: the Clarifier asks which symbol; the reply is charted (~15s). It asks at most once per conversation. |

## Data Sources

### Loading the Reference Database

`data/refdata.duckdb` holds **625 instruments** (780 KB): all **503** S&P 500 constituents plus
**122** hand-curated ETFs in 15 families (index trackers, sector SPDRs, bonds, commodities, factor
funds, and the leveraged/inverse families). It ships inside the container image, so the deployed app
needs no setup.

Nothing needs loading — read it directly:

```python
import duckdb
rows = duckdb.connect("data/refdata.duckdb", read_only=True).execute(
    "SELECT symbol, name, gics_sector FROM instruments WHERE gics_sector = 'Energy'"
).fetchall()
```

To refresh membership, run `uv run build/build_refdata.py` (under a minute). It rebuilds the
database from Wikipedia's S&P 500 table, the SEC's ticker file and the Nasdaq Trader directory, and
writes `data/refdata_mappings.json`, the schema card the Text2SQL tools give the model.

Prices are not stored; every statistic is computed live from Nasdaq's end-of-day bars.

### The Feeds

| Source | What it provides | Freshness |
|---|---|---|
| Nasdaq (`api.nasdaq.com`) | Daily OHLCV for US stocks and ETFs, 10 years, split-adjusted | End of day |
| Yahoo Finance (`query1.finance.yahoo.com`) | The same bars, used only when Nasdaq fails | End of day |
| SEC EDGAR | 8-K events with item codes, 10-Ks and 10-Qs | Same day |
| Google News RSS | Headlines for a symbol or keywords | Same day |
| Nasdaq Trader | Symbol directory: listings, names, ETF flags | Daily |
| Wikipedia | S&P 500 constituents with GICS (offline build only) | Monthly, in practice |

All six are free public feeds and need no API keys. Every feed is called with one generic
User-Agent carrying no personal contact details.

## Tools

| Tool | Purpose | Source |
|---|---|---|
| `screen_pairs` | Fit and rank every pair in a universe (sector, ETF family or list; 40 names max), then holdout-backtest the top 8 | Nasdaq + reference DB |
| `analyze_pair` | One pair in full: statistics, holdout backtest, divergence episodes, z-score chart | Nasdaq |
| `research_divergence` | SEC filings for both legs over one divergence window, and news over its first 14 days | Google News, SEC EDGAR |
| `get_reference_schema` | The reference table's columns and real enum values, for writing SQL | Reference DB |
| `query_reference` | One read-only SELECT; returns the SQL it ran, for citation | Reference DB |
| `get_price_history` | Period return, high, low, realized volatility, last five closes, month-end table | Nasdaq |
| `get_chart` | Interactive daily closing-price line chart, on request | Nasdaq |
| `get_news` | Headlines for a symbol or keywords: last 7 days, or a range up to 14 | Google News |
| `get_filings` | 8-K events and 10-K/10-Q reports: latest, or a date range | SEC EDGAR |

## API

`POST /chat` takes `{"message", "session_id"?, "model"?}` and keeps Project 1's response shape, with
two additions: each tool call carries the `agent` that made it and its timing, and `charts` carries
the figures for the page. A researcher's calls are tagged with its own name, so the fan-out is
visible in the JSON alone. Steps that call no tool get rows too (`clarify`, `agent`, `researcher`,
`writer`), so the whole run can be drawn from `tool_calls`. A failed step or tool call carries an
`error`, and a retry that replaced it carries `"retry": true` in its `args`.

```json
{
  "response": "The evidence does not support … [F1.1.2]",
  "session_id": "4f0c…",
  "tool_calls": [
    {"name": "screen_pairs", "args": {"sector": "Energy"}, "result": "{…}",
     "agent": "Researcher 1: Screen the energy sector…", "started_at": 14.2, "ended_at": 23.9}
  ],
  "charts": [{"symbol": "…", "figure": {…}, "agent": "…"}]
}
```

## Files

```
app.py                   FastAPI only: the page, /chat, /clear, the price tape, the Plotly bundle
config.py                Models and fallbacks, run caps and timeouts, screening and backtest rules
index.html               The UI

pipeline/orchestrator.py The seam between the web layer and the agents
pipeline/roster.py       The five agents and every instruction they are given
pipeline/research.py     The asyncio.gather fan-out, and the Writer that follows it
pipeline/citations.py    The Python citation audit
pipeline/schemas.py      The structured hand-offs between agents
pipeline/runtime.py      Per-run state the model never sees
pipeline/tools.py        The nine tools: plain Python bodies, @function_tool wrappers
pipeline/chart.py        Themed Plotly specs: lines, tables, the z-score panel

quant/pairs.py           Hedge ratio, spread, z-scores, half-life, rolling beta
quant/backtest.py        Holdout split, frozen parameters, deflated Sharpe
quant/screen.py          Every combination in a universe, fitted and ranked

sources/data.py          Nasdaq, Yahoo Finance, SEC EDGAR, Google News, Nasdaq Trader
sources/refdata.py       The read-only DuckDB connection and universe resolution

data/                    The committed reference database and its schema card
build/                   Offline, run once, never imported at request time

pyproject.toml, uv.lock  Dependencies, pinned; `uv sync` installs them
submission.json          The deploy URL, author and path, for grading
```

## Run Locally

Requires a Google Cloud project with Vertex AI enabled.

```bash
gcloud auth application-default login
uv run app.py
```

Then open http://localhost:8000.

If it cannot resolve the project, set `GOOGLE_CLOUD_PROJECT=<your project id>`.

## Cloud Deployment

- **Main:** https://tool-calling-agent-git-llmb6vmaoa-nn.a.run.app
- **Backup:** https://tool-calling-agent-git-401765433529.northamerica-northeast1.run.app

**Sign in with your Columbia Google account when prompted; if access is refused, email
jay2147@columbia.edu.**

Hosted on Google Cloud Run behind Identity-Aware Proxy. A Cloud Build trigger builds each merge to
`main` on GitHub with Google Cloud buildpacks and supplies the entrypoint as a substitution. These
settings live in the service rather than in version control, so they are recorded here:

```
_ENTRYPOINT = uvicorn app:app --host 0.0.0.0 --port $PORT
_SERVICE_NAME = tool-calling-agent-git
```

| Setting | Value | Reason |
|---|---|---|
| `--timeout` | 900s | The 300s default is exactly the grading window, so a slow screen would race the proxy |
| `--memory` | 1Gi | 307 MB at idle from imports, ~400 MB during a screen; the 512 MB default leaves no room |
| `--cpu` | 2 | Fitting every pair is CPU-bound: the 40-name cap is 780 regressions plus rolling betas |
| `--min-instances` | 1 | Keeps one warm instance so chat history survives between requests. Bills continuously |
| `--max-instances` | 1 | Sessions live in one instance's memory, so a second instance would not see them |

The reference database ships inside the image, opened read-only. Cloud Run's filesystem is in
memory, so conversation history (`conversations.db`) is per-instance and is lost when the instance
restarts or a new revision is deployed.

## Future Improvements

### Showing the Work While It Happens

- **Live progress during the wait.** Fill in the run timeline as each step happens during the
  two-minute wait, rather than drawing it only once the answer arrives.
- **SSE streaming.** A `GET /stream` side channel that sends each step as it happens, while
  `POST /chat` keeps owning the run so closing the tab cannot kill it.
- **Researcher columns.** Each researcher's brief, tools and findings side by side while they work.

### Agent Architecture

- **A Critic, and an iterative review cycle.** An agent that attacks results that look too good — a
  high Sharpe, a short holdout, a clean equity curve — and sends the Lead back for another round. It
  is the natural next agent shape, and it fits the overfitting theme exactly.
- **Multi-turn clarification.** The Clarifier asks one question by design so it cannot loop; a
  richer version would ask a short series: universe, then window, then what "good" means.
- **`[F…]` ids on the fast path.** The analyst path lists its evidence and flags unverified numbers
  but has no findings store, so it cannot cite each number the way the research path does.

### Evaluation

- **An eval suite.** Test discipline rather than accuracy: a vague question is clarified exactly
  once, no number appears uncited, a fabricated finding is caught, and a random-walk pair is never
  called tradeable. Ablations (no fan-out, no gap loop, no citation audit) would then show the
  architecture earns its complexity.

### Data and Analysis

- **More granular pricing.** Intraday history for in-session entry and exit timing, realistic
  slippage, and whether a divergence opened on one print or over hours.
- **Basket vs basket.** Cointegrated baskets on both sides, including sparse replication of a sector
  ETF by a handful of its constituents.
- **Futures legs.** Basis trades such as VIX futures against VXX, or ES against SPY.
- **Dividend-adjusted returns.** Total returns in place of today's price returns.
- **Persistent sessions.** Several conversations at once, surviving a restart, which needs storage
  outside the container's in-memory filesystem.
