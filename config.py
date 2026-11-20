"""Model choice per role and the caps on one run.

Set so the worst case finishes inside 3.5 minutes. Data limits live in pipeline/tools.py.
"""

# --- Models ---

# Pinned for the Clarifier, Lead and Writer (they ignore the dropdown). The Lead's planning turn
# measured 6-14s here against 8-68s on 3.8 Flash (2026-11-19).
REASONING_MODEL = "vertex_ai/gemini-3.5-flash"

# Dropdown name -> LiteLLM id: the GA 3.x models Vertex serves this project (probed 2026-11-17).
RESEARCHER_MODELS = {
    "Gemini 3.5 Flash-Lite": "vertex_ai/gemini-3.5-flash-lite",
    "Gemini 3.8 Flash": "vertex_ai/gemini-3.8-flash",
    "Gemini 3.5 Flash": "vertex_ai/gemini-3.5-flash",
    "Gemini 3.1 Flash-Lite": "vertex_ai/gemini-3.1-flash-lite",
}
# Flash-Lite: measured 2-3x faster than Flash on the same questions.
DEFAULT_RESEARCHER_MODEL = "Gemini 3.5 Flash-Lite"

# Pinned so reasoning does not vary by model (unset, the Flash-Lites do none).
REASONING_EFFORT = "medium"

# Retried on when a pinned agent fails (10/10 gate verdicts; reports in 10-12s).
FALLBACK_MODEL = "vertex_ai/gemini-3.5-flash-lite"

# --- Model calls ---

# Per model call, and LiteLLM retries a timeout. 25s clears the 95th percentile of every agent turn
# in the recorded runs (Analyst 20.9s, Researcher 16.9s, Lead planning at most 13.9s).
TURN_TIMEOUT_S = 25
WRITER_CALL_TIMEOUT_S = 45  # 3.5 Flash wrote reports in 22-28s; a stall moves to FALLBACK_MODEL
MODEL_RETRIES = 2  # Vertex rate-limits bursts, and a fan-out makes many calls at once
PINNED_RETRIES = 0  # a stalled Lead or Writer moves to FALLBACK_MODEL instead of retrying twice

# Held back for the Writer (25-35s), including WRITER_FALLBACK_S for its retry.
WRITEUP_RESERVE_S = 75

# Retrying into an exhausted quota spends the budget; after this many, stop and report.
MAX_RATE_LIMITS = 3

# --- Run budgets: caps live here, not in prompts, so a model cannot negotiate them ---

MAX_TURNS = 12  # one turn is a model call plus the tool calls it asks for, so this caps rounds
# A failed gate would send a vague question to research, so it is retried on FALLBACK_MODEL.
CLARIFIER_TIMEOUT_S = 12  # 3.5 Flash: 90% of 30 gate calls under 8s; 2 took 15s and 48s
CLARIFIER_FALLBACK_TIMEOUT_S = 10  # Flash-Lite: 95% under 8.5s, slowest of 30 was 9.2s
LEAD_TURNS = 6  # enough for research -> research -> write_report, with slack
MAX_FANOUT_ROUNDS = 2  # a second round exists to fill a stated gap, not to keep fishing
MAX_BRIEFS_PER_ROUND = 4
MAX_CONCURRENT_RESEARCHERS = 4  # Vertex 429s under heavier fan-out
MAX_RESEARCHER_TURNS = 12  # each tool call is a turn, so this must exceed the tool budget below
MAX_RESEARCHER_TOOL_CALLS = 6  # quoted in the prompt; the turn cap above is the hard backstop
RESEARCHER_TIMEOUT_S = 110  # a sector-screen researcher measures 15-25s; the rest is fetch slack
RESEARCHER_SALVAGE_S = 30  # a timed-out researcher's write-up of what it already gathered
MIN_RESEARCHER_S = 20  # shortest research time worth starting a round for
ROUND_MARGIN_S = 5  # a round returns at least this long before the main run is cut off
WRITER_TIMEOUT_S = 90  # still capped by what is left of TOTAL_BUDGET_S when the Writer starts
WRITER_FALLBACK_S = 25  # kept back from the first Writer attempt for a retry on FALLBACK_MODEL
# Every stage is timed against what is left of TOTAL_BUDGET_S, so a run ends by about 205s.
TOTAL_BUDGET_S = 200

# --- Screening ---

# Long enough for a half-life to mean anything, short enough to exclude the pre-pandemic regime.
LOOKBACK_DAYS = 3 * 365
MIN_LOOKBACK_DAYS = 400  # must leave a holdout worth reporting after the in-sample split
MAX_LOOKBACK_DAYS = 10 * 365  # roughly what the price feed serves
MIN_OVERLAP_SESSIONS = 250  # a pair needs this much shared history to be screened
MIN_BACKTEST_SESSIONS = 120  # floor for analysing one pair directly

SCREEN_THREADS = 4  # per symbol, not per pair; data._SLOTS caps the real concurrency anyway
TOP_K = 8  # pairs carried from the screen into a full backtest
MAX_TOP_K = 20
MAX_UNIVERSE = 40  # 40 names is already 780 pairs; screening is quadratic in this

# Every metric is ranked first, so one outlier cannot dominate. A weight of 0 ignores a metric.
SCREEN_WEIGHTS = {
    "correlation": 1.0,
    "half_life_days": 1.0,
    "crossings_per_year": 1.0,
    "beta_stability": 1.0,
}

# True where a bigger raw value ranks better. Must name every metric in SCREEN_WEIGHTS.
SCREEN_HIGHER_IS_BETTER = {
    "correlation": True,
    "half_life_days": False,
    "crossings_per_year": True,
    "beta_stability": False,
}

# --- Backtest ---

# Defaults for backtest.Rules, adjustable per request.
IN_SAMPLE_FRAC = 0.70  # chronological, never random
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0
MAX_HOLD_SESSIONS = 60

# A spread that cannot halve within one maximum hold is dropped (a random walk shows a long one).
MAX_HALF_LIFE_DAYS = float(MAX_HOLD_SESSIONS)

COST_BPS = 5.0  # per leg, per fill
BORROW_BPS_ANNUAL = 50.0  # on the short leg

# --- Divergence episodes ---

DIVERGENCE_Z = 2.0
DIVERGENCE_MIN_DAYS = 10
MAX_EPISODES_RESEARCHED = 3  # news and filings are fetched per episode, so this caps the fan-out
