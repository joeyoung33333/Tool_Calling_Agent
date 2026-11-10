"""Reference values: which model each role uses, and the caps on one run.

These are starting values, not commitments: real latency is unknown until the agent is timed end to
end on Cloud Run, so they get re-tuned against a measured p95 before submission.

Data limits (history length, news window, chart points) live beside the tools that enforce them,
in tools.py.
"""

# --- Models ---

# Whichever agent reasons or writes prose runs on Flash: Phase 3's Lead must emit parallel tool calls, which Flash-Lite botches.
REASONING_MODEL = "vertex_ai/gemini-3.5-flash"

# UI name -> LiteLLM id for the model the user picks. Researchers carry the cost/quality trade-off: several, each multi-turn.
RESEARCHER_MODELS = {
    "Gemini 3.5 Flash-Lite": "vertex_ai/gemini-3.5-flash-lite",
    "Gemini 2.5 Flash-Lite": "vertex_ai/gemini-2.5-flash-lite",
    "Gemini 3.5 Flash": "vertex_ai/gemini-3.5-flash",
    "Gemini 2.5 Flash": "vertex_ai/gemini-2.5-flash",
}
DEFAULT_RESEARCHER_MODEL = "Gemini 3.5 Flash-Lite"

# --- Model calls ---

MODEL_TIMEOUT_S = 90  # a stalled call would otherwise hang the page
MODEL_RETRIES = 2  # Vertex rate-limits bursts, and a fan-out makes many calls at once

# --- Turn budgets ---

# One turn is a model call plus every tool call it asks for, so this caps rounds, not calls.
MAX_TURNS = 12

# --- Screening ---

# How much history a screen looks at. Cboe carries daily bars from 2004, so 3 years is always
# available for an established name; a leveraged ETF starts at its own inception. Three years is
# long enough for a half-life and a rolling beta to mean anything, and short enough that a
# pre-pandemic regime is not driving the correlation.
LOOKBACK_DAYS = 3 * 365
# Floor and ceiling for a caller-supplied lookback. The floor has to leave a holdout worth reporting
# after the in-sample split; the ceiling is roughly what the price feed actually serves.
MIN_LOOKBACK_DAYS = 400
MAX_LOOKBACK_DAYS = 10 * 365
MIN_OVERLAP_SESSIONS = 250  # fewer shared sessions than this and the statistics are not worth citing

SCREEN_THREADS = 6  # Cboe fetches are per symbol, not per pair, and it throttles bursts
TOP_K = 8  # pairs carried from the screen into a full backtest
MAX_TOP_K = 20
MAX_UNIVERSE = 40  # 40 names is already 780 pairs; screening is quadratic in this

# What makes a pair worth backtesting, before any backtest has run. Equal weights by default, and
# every metric is turned into a rank first, so one pair with a freak half-life cannot dominate.
# Raise a weight to care more about it; set one to 0 to ignore that metric.
# True where a bigger raw value ranks better. Must name every metric in SCREEN_WEIGHTS: the screen
# refuses to guess a direction, because guessing wrong silently inverts the ranking.
SCREEN_HIGHER_IS_BETTER = {
    "correlation": True,
    "half_life_days": False,
    "crossings_per_year": True,
    "beta_stability": False,
}

SCREEN_WEIGHTS = {
    "correlation": 1.0,  # higher is better
    "half_life_days": 1.0,  # shorter is better, and a pair with no half-life is dropped
    "crossings_per_year": 1.0,  # more round trips is better
    "beta_stability": 1.0,  # lower is better: one relationship, not a different one each regime
}

# --- Backtest ---

# Defaults for backtest.Rules. Adjustable per request; these are what an unqualified question gets.
IN_SAMPLE_FRAC = 0.70  # chronological, never random
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0
MAX_HOLD_SESSIONS = 60
# A spread that needs longer than one maximum holding period to decay halfway back cannot complete a
# round trip inside a trade, so the screen drops it. This is a real gate: OLS on a random walk is
# biased toward a slightly negative slope, so a wandering spread reports a long finite half-life
# rather than none at all, and without a ceiling it would rank as if it mean-reverted.
MAX_HALF_LIFE_DAYS = float(MAX_HOLD_SESSIONS)
COST_BPS = 5.0  # per leg, per fill
BORROW_BPS_ANNUAL = 50.0  # on the short leg

# --- Divergence episodes ---

DIVERGENCE_Z = 2.0
DIVERGENCE_MIN_DAYS = 10
MAX_EPISODES_RESEARCHED = 3  # news and filings are fetched per episode, so this caps the fan-out
