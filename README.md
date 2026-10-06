# Ticker: A Markets Research Agent

IEOR-E4570: Project 1

Author: Joseph Young (jay2147)

## Overview

A chat agent for market research and analysis of US equities. It calls tools for live and historical prices, options, news headlines and SEC filings, then answers with prices, volatility, options-implied moves, news, and filings that may explain a price move. Every answer cites its source and data delay.

For research only, not investment advice. Always double-check results.

### Design Choices

- **Agent:** Gemini on Vertex AI via LiteLLM, in a tool-calling loop.
- **Settings:** The UI allows model selection (four Gemini models) and setting a tool call limit of 1 to 10 (default 5). At the limit, the agent answers with what it has, and the UI shows a red limit-reached notice.
- **Tools:** 6 tools backed by external data sources (Cboe, Google News, SEC EDGAR), using free public feeds with no API keys. Tools include descriptions, argument examples, and actionable error messages.
- **Explaining Moves:** The agent lines up price history with news and SEC filings around the date, then judges which items could plausibly move the stock.
- **Charts:** Drawn only on request. `get_chart` builds an SVG in `chart.py`; the UI shows it and the model only sees a short text summary.
- **Analysis:** Price returns, volatility and implied moves are computed in Python, not by the model.
- **Memory:** Full message history is kept in memory per `session_id`, so follow-ups work and sessions stay separate; it resets on restart or New Session.
- **UI:** A modern, minimal take on a Bloomberg terminal (black, amber, monospace) with clickable sample queries.

### Market Universe

**Equities**

- US stocks and ETFs (~15 min delayed; the last close when the market is closed)
- US indices: SPX, NDX, RUT and the VIX level (~15 min delayed)
- US listed options on stocks, ETFs and indices (~15 min delayed)

**News and Filings**

- News headlines for any ticker or keywords, latest or past dates (Google News; headlines only, no article text)
- SEC filings for US-listed companies: 8-K events, 10-K and 10-Q (not ETFs or indices)

**Charts**

- Daily closing-price line charts for any supported symbol, on request

**Not covered:** crypto, VIX and CME futures, executive share transactions (Form 4), fundamentals, rates and FX.

## Cloud Deployment

Deployed on Google Cloud Run, which rebuilds automatically on every push to `main`. Both links reach the same service; sign in with a Columbia Google account.

- Main: https://tool-calling-agent-git-llmb6vmaoa-nn.a.run.app
- Backup: https://tool-calling-agent-git-401765433529.northamerica-northeast1.run.app

## Run Locally

Requires a Google Cloud project with Vertex AI enabled.

```bash
gcloud auth application-default login
uv run app.py
```

Then open http://localhost:8000. If it can't resolve the project, set `GOOGLE_CLOUD_PROJECT=<your project id>`.

## Sample Queries

1. "What's NVDA trading at, and what is the options market implying for its next-expiry move?" (price and options)
2. "TSLA jumped on October 2. What happened?" (price history, news and SEC filings)
3. "Chart AAPL's price over the last 6 months." (chart)

## Tools

| Tool | Purpose | Source |
|---|---|---|
| `get_price` | Latest price, change, bid/ask, volume, 30-day IV | Cboe |
| `get_price_history` | Return, high/low, realized vol over a date range | Cboe |
| `get_chart` | Daily closing-price line chart, on request | Cboe |
| `get_option_chain` | Strikes near the money, ATM IV, implied move, put/call ratios | Cboe |
| `get_news` | Headlines for a ticker or keywords: last 7 days, or a date range | Google News |
| `get_filings` | SEC 8-K events and 10-K/10-Q reports: latest, or a date range | SEC EDGAR |

## Future Improvements

**Market Data**

- Crypto: spot, futures, perpetuals and funding rates, options
- Futures: VIX futures curve and CME index futures (ES, NQ), which need a keyed data vendor
- Rates (SOFR, Treasury yield curve) and FX with implied forwards
- Form 4 filings (share transactions by executives and directors)
- Full article text for news

**Analysis**

- Further analysis tools: risk metrics (volatility, drawdown, VaR, beta), correlations and comparisons
- Richer options outputs: skew, term structure, open-interest strikes
- Backtest simulation: run a trading strategy over historical prices and report returns and drawdown
- Order book display (depth, spread, imbalance) and slippage estimates

**Charts**

- More chart types (comparisons, curves)

## Files

- `app.py`: harness loop, sessions, `/chat`, `/clear`
- `tools.py`: tool schemas, tool logic, `run_tool()`
- `chart.py`: draws line charts as SVG
- `data.py`: Cboe, Google News and SEC EDGAR fetchers
- `index.html`: UI
