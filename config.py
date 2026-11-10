"""Which model each role uses, and the caps on one run.

Set from measured runs: a sector screen lands at 25-35s against the 240s budget.

Data limits (history length, news window, chart points) live beside the tools that enforce them,
in tools.py.
"""

# --- Models ---

# The Clarifier's model, and the fallback before the caller's choice is applied. Flash rather than
# Flash-Lite because the gate runs on every question and its verdict decides the route.
REASONING_MODEL = "vertex_ai/gemini-3.5-flash"

# UI name -> LiteLLM id for the model the user picks.
RESEARCHER_MODELS = {
    "Gemini 3.5 Flash-Lite": "vertex_ai/gemini-3.5-flash-lite",
    "Gemini 2.5 Flash-Lite": "vertex_ai/gemini-2.5-flash-lite",
    "Gemini 3.5 Flash": "vertex_ai/gemini-3.5-flash",
    "Gemini 2.5 Flash": "vertex_ai/gemini-2.5-flash",
}
# Flash-Lite. Flash was tried as the default on the argument that a 240s budget left headroom for a
# stronger model, but measurement killed it: the same sector screen took 81s on Flash against
# 12-22s on Flash-Lite, and a single-pair question 38s against ~15s. A fast path advertised as
# answering "in seconds" that takes 38 is a worse product than a slightly weaker one that takes 15.
# Flash is one click away in the dropdown for anyone who wants it on a hard question.
DEFAULT_RESEARCHER_MODEL = "Gemini 3.5 Flash-Lite"

# --- Model calls ---

# A healthy call measures 1-3s, so 45s is already far past anything legitimate. It matters that
# TIMEOUT x (1 + RETRIES) stays under TOTAL_BUDGET_S: at 90s x 3 one stalled call could burn 270s
# of a 240s budget on its own, which is how a 30s question became a 200s one.
MODEL_TIMEOUT_S = 45
MODEL_RETRIES = 2  # Vertex rate-limits bursts, and a fan-out makes many calls at once

# Held back from the main run so there is always time to write up what was gathered. Without it a
# run that spends the whole budget researching has nothing left to answer with.
WRITEUP_RESERVE_S = 45

# Rate limits are not transient when the quota is simply gone: retrying into them spends the budget
# and returns nothing. After this many, stop starting work and report what we have.
MAX_RATE_LIMITS = 3

# --- Run budgets ---
#
# Every cap lives here rather than in a prompt. A model asked to respect its own budget will
# sometimes not, and the failure is invisible; a number checked in Python cannot be negotiated.

MAX_TURNS = 12  # one turn is a model call plus the tool calls it asks for, so this caps rounds
CLARIFIER_TIMEOUT_S = 20  # a gate that is slow to say 'go ahead' is worse than no gate
LEAD_TURNS = 6  # enough for research -> research -> write_report, with slack
MAX_FANOUT_ROUNDS = 2  # a second round exists to fill a stated gap, not to keep fishing
MAX_BRIEFS_PER_ROUND = 4
MAX_CONCURRENT_RESEARCHERS = 4  # Vertex 429s under heavier fan-out
MAX_RESEARCHER_TURNS = 12  # each tool call is a turn, so this must exceed the tool budget below
MAX_RESEARCHER_TOOL_CALLS = 6  # quoted in the prompt; the turn cap above is the hard backstop
RESEARCHER_TIMEOUT_S = 110  # a sector screen measures ~15s; the rest is slack for a slow fetch
WRITER_TIMEOUT_S = 60
TOTAL_BUDGET_S = 240  # inside Cloud Run's 300s default even if our 900s setting is ever lost

# --- Screening ---

# Three years is long enough for a half-life and a rolling beta to mean anything, and short enough
# that a pre-pandemic regime is not driving the correlation.
LOOKBACK_DAYS = 3 * 365
MIN_LOOKBACK_DAYS = 400  # must leave a holdout worth reporting after the in-sample split
MAX_LOOKBACK_DAYS = 10 * 365  # roughly what the price feed serves
MIN_OVERLAP_SESSIONS = 250  # a pair needs this much shared history to be screened
MIN_BACKTEST_SESSIONS = 120  # the lower floor for analysing one pair directly, below which the statistics are not worth citing

SCREEN_THREADS = 4  # per symbol, not per pair; data._SLOTS caps the real concurrency anyway
TOP_K = 8  # pairs carried from the screen into a full backtest
MAX_TOP_K = 20
MAX_UNIVERSE = 40  # 40 names is already 780 pairs; screening is quadratic in this

# What makes a pair worth backtesting, before any backtest has run. Every metric becomes a rank
# first, so one pair with a freak half-life cannot dominate. Set a weight to 0 to ignore a metric.
SCREEN_WEIGHTS = {
    "correlation": 1.0,
    "half_life_days": 1.0,
    "crossings_per_year": 1.0,
    "beta_stability": 1.0,
}

# True where a bigger raw value ranks better. Must name every metric in SCREEN_WEIGHTS: the screen
# refuses to guess a direction, because guessing wrong silently inverts the ranking.
SCREEN_HIGHER_IS_BETTER = {
    "correlation": True,
    "half_life_days": False,
    "crossings_per_year": True,
    "beta_stability": False,
}

# --- Backtest ---

# Defaults for backtest.Rules: what an unqualified question gets. Adjustable per request.
IN_SAMPLE_FRAC = 0.70  # chronological, never random
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0
MAX_HOLD_SESSIONS = 60

# A spread needing longer than one maximum hold to decay halfway back cannot complete a round trip,
# so the screen drops it. This is a real gate, not a formality: OLS on a random walk is biased
# toward a slightly negative slope, so a wandering spread reports a long finite half-life rather
# than none, and without a ceiling it would rank as if it mean-reverted.
MAX_HALF_LIFE_DAYS = float(MAX_HOLD_SESSIONS)

COST_BPS = 5.0  # per leg, per fill
BORROW_BPS_ANNUAL = 50.0  # on the short leg

# --- Divergence episodes ---

DIVERGENCE_Z = 2.0
DIVERGENCE_MIN_DAYS = 10
MAX_EPISODES_RESEARCHED = 3  # news and filings are fetched per episode, so this caps the fan-out
