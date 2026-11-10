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

## Status

Phase 1 is complete: the agent runs on the OpenAI Agents SDK with five tools, SQLite-backed
sessions and interactive Plotly output. The pair screening, backtest and multi-agent research
fan-out described above are Phases 2–4 and are **not built yet**. This README describes what the
code does today; the final version records the full pipeline, sample queries and evaluation.

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
- **Analysis in Python:** returns, volatility, drawdown, VaR, Sharpe, Sortino, beta and alpha are
  computed in Python, not by the model.
- **Charts:** one library, Plotly, for every visualization — lines and tables today, spread and
  correlation views next. Figures are built in Python as specs and rendered by the page, so they are
  interactive (zoom, pan, hover readout, drag to compare dates) and **never enter the model's
  context**, which would otherwise replay thousands of tokens on every later turn. Plotly is served
  from the installed package, not a CDN, so a blocked CDN cannot leave a grader with no charts.
- **UI:** a modern take on a Bloomberg terminal (black, amber, monospace) with a live price tape and
  clickable sample queries.

## Data Sources

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
