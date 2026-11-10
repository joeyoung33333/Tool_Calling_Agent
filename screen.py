"""Screen every pair in a universe, then rank them. The model never chooses a pair.

This is the module the whole project turns on. An agent decides *what to look at* - a sector, an ETF
family, a hunch - and interprets what comes back. Everything between those two points happens here,
in Python, over every combination:

    n symbols  ->  n price fetches  ->  C(n,2) pair fits  ->  rank  ->  top K backtested

The asymmetry is the point. Fetching is linear and slow, so each series is pulled once and reused;
pair arithmetic is quadratic but nearly free over cached series. A 40-name universe is 40 fetches
and 780 fits. Because the ranking is a deterministic function of the statistics, the answer to "why
this pair?" is a number, not a sentence a model wrote.

Ranking is done on ranks, not raw values, so one pair with a freak half-life cannot dominate the
composite. Weights live in config.SCREEN_WEIGHTS.
"""

import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from itertools import combinations

import backtest
import config
import data
import pairs


class ScreenError(Exception):
    """Something the caller can fix: too few symbols, no usable history."""


def fetch_series(symbols: list[str], lookback_days: int) -> tuple[dict[str, list[dict]], list[dict]]:
    """Daily bars for each symbol, fetched once. Returns (series, problems).

    Per symbol, not per pair: this is the only linear-cost step, and every pair that follows reads
    these same lists. A symbol that fails is reported rather than silently dropped, because a
    missing leg changes which pairs were even considered.
    """
    start = date.today() - timedelta(days=lookback_days)
    series: dict[str, list[dict]] = {}
    problems: list[dict] = []

    def one(symbol: str) -> None:
        try:
            bars = data.price_history(symbol, start)
        except Exception as e:
            problems.append({"symbol": symbol, "problem": str(e)})
            return
        if len(bars) < config.MIN_OVERLAP_SESSIONS:
            problems.append({"symbol": symbol,
                             "problem": f"only {len(bars)} sessions of history, need {config.MIN_OVERLAP_SESSIONS}"})
            return
        series[symbol] = bars

    with ThreadPoolExecutor(config.SCREEN_THREADS) as pool:
        list(pool.map(one, symbols))
    return series, problems


def clamp_lookback(days: int | None) -> int:
    """`days` of history to request, held inside a sane range whatever the caller asked for."""
    return min(max(int(days or config.LOOKBACK_DAYS), config.MIN_LOOKBACK_DAYS), config.MAX_LOOKBACK_DAYS)


def fit_pair(a: str, b: str, series: dict[str, list[dict]], rules: backtest.Rules) -> dict | None:
    """Statistics for one pair, measured on the in-sample window only, or None if incomparable.

    In-sample only, and that is the whole point. Screening on the full window would choose which
    pair to report using the very sessions the backtest then calls a holdout: a pair that mean-
    reverts only in the last 30% would be selected *because* of that and would then pass its own
    test. Freezing the parameters is not enough if the selection saw the answer.

    Orientation is fixed alphabetically so a pair is screened once, not twice, and so the reported
    hedge ratio always means "units of the second symbol per unit of the first".
    """
    dates, closes_a, closes_b = pairs.align(series[a], series[b])
    if len(dates) < config.MIN_OVERLAP_SESSIONS:
        return None
    cut = backtest.split(dates, rules)
    try:
        stats = pairs.summary(dates[:cut], closes_a[:cut], closes_b[:cut])
        model = backtest.fit(dates[:cut], closes_a[:cut], closes_b[:cut])
    except (ValueError, statistics.StatisticsError):
        return None
    # Where the spread sits *today*, scored against the in-sample fit rather than its own window.
    # summary() measured z over the in-sample slice, so its last value is months stale here.
    current = (closes_a[-1] - model.beta * closes_b[-1] - model.mean) / model.sd
    return {"a": a, "b": b, **stats, "current_z": round(current, 2), "current_z_as_of": dates[-1],
            "fitted_on_sessions": cut, "holdout_sessions": len(dates) - cut}


def rank(candidates: list[dict], weights: dict[str, float] | None = None) -> list[dict]:
    """Order `candidates` best-first on a weighted average of per-metric ranks.

    Ranks rather than z-scores or raw values: a half-life of 2 days and one of 400 are both
    outliers, and on raw values either would swamp every other metric. Each candidate gets
    `rank_score` (0 worst, 1 best) and `rank_detail` so the ordering can be explained exactly.
    """
    weights = weights or config.SCREEN_WEIGHTS
    reverting = [c for c in candidates if c.get("tradeable") and _reverts(c)]
    if not reverting:
        return []

    # Position in `reverting`, not id(), so a copied candidate cannot silently score as missing.
    scores: dict[str, dict[int, float]] = {}
    for metric, weight in weights.items():
        if weight <= 0:
            continue
        if metric not in config.SCREEN_HIGHER_IS_BETTER:
            raise ScreenError(
                f"No ranking direction is defined for '{metric}'. Add it to "
                f"config.SCREEN_HIGHER_IS_BETTER alongside its weight."
            )
        have = [i for i, c in enumerate(reverting) if c.get(metric) is not None]
        have.sort(key=lambda i: reverting[i][metric],
                  reverse=not config.SCREEN_HIGHER_IS_BETTER[metric])
        # Worst gets 0, best gets 1. One candidate is the best there is, so it gets 1, not 0.
        last = len(have) - 1
        scores[metric] = {i: (position / last if last else 1.0) for position, i in enumerate(have)}

    for i, c in enumerate(reverting):
        detail, total, weighed = {}, 0.0, 0.0
        for metric, weight in weights.items():
            if weight <= 0:
                continue
            # A missing metric scores the middle rather than being dropped. Dropping it renormalised
            # the average over a smaller weight set, so a candidate that *had* a value scored on it
            # while its rivals did not - making extra data a handicap and inverting the ranking.
            score = scores[metric].get(i, 0.5)
            detail[metric] = round(score, 3)
            total += weight * score
            weighed += weight
        c["rank_score"] = round(total / weighed, 4) if weighed else 0.0
        c["rank_detail"] = detail
    return sorted(reverting, key=lambda c: c["rank_score"], reverse=True)


def screen(
    symbols: list[str],
    lookback_days: int | None = None,
    top_k: int | None = None,
    rules: backtest.Rules | None = None,
) -> dict:
    """Fit every pair in `symbols`, rank them, and backtest the best `top_k`.

    `trials_considered` is counted here, in code, and handed to the deflated Sharpe: the haircut for
    having searched many pairs cannot be negotiated down by whatever is reading the result.
    """
    wanted = sorted({s.strip().upper() for s in symbols if s and s.strip()})
    if len(wanted) < 2:
        raise ScreenError("Screening needs at least 2 symbols.")
    if len(wanted) > config.MAX_UNIVERSE:
        raise ScreenError(f"{len(wanted)} symbols is over the {config.MAX_UNIVERSE} cap "
                          f"({len(wanted) * (len(wanted) - 1) // 2} pairs). Narrow the universe.")
    lookback = clamp_lookback(lookback_days)
    k = min(max(1, top_k or config.TOP_K), config.MAX_TOP_K)
    rules = rules or backtest.Rules()

    series, problems = fetch_series(wanted, lookback)
    if len(series) < 2:
        raise ScreenError(
            "Fewer than 2 symbols returned usable history, so there is nothing to pair. "
            + "; ".join(f"{p['symbol']}: {p['problem']}" for p in problems[:4])
        )

    with_history = sorted(series)
    combos = list(combinations(with_history, 2))
    candidates = [fit for a, b in combos if (fit := fit_pair(a, b, series, rules))]
    ranked = rank(candidates)
    rejected = [c for c in candidates if not c.get("tradeable")]
    no_reversion = [c for c in candidates if c.get("tradeable") and not _reverts(c)]

    # Only the top K pay for a backtest; the arithmetic above already eliminated the rest.
    tested = []
    for candidate in ranked[:k]:
        try:
            result = backtest.evaluate(
                series[candidate["a"]], series[candidate["b"]], rules=rules,
                trials=len(candidates),
            )
        except (ValueError, statistics.StatisticsError) as e:
            tested.append({**candidate, "backtest": {"tradeable": False, "reason": str(e)}})
            continue
        tested.append({**candidate, "backtest": result})

    return {
        "universe": with_history,
        "universe_size": len(with_history),
        "pairs_combinations": len(combos),
        "pairs_screened": len(candidates),
        "pairs_skipped_short_history": len(combos) - len(candidates),
        "trials_considered": len(candidates),
        "lookback_days": lookback,
        "top_k": k,
        "ranked": tested,
        "also_ranked": [_brief(c) for c in ranked[k:]],
        "rejected_untradeable": [{"a": c["a"], "b": c["b"], "why": c["not_tradeable_because"]} for c in rejected[:10]],
        "rejected_no_mean_reversion": [
            {"a": c["a"], "b": c["b"], "half_life_days": c["half_life_days"]} for c in no_reversion[:10]
        ],
        "mean_reversion_gate": (
            f"A pair is screened only if its spread has a measurable half-life of at most "
            f"{config.MAX_HALF_LIFE_DAYS:.0f} sessions, the maximum holding period. "
            f"{len(no_reversion)} of {len(candidates)} tradeable pairs failed that test."
        ),
        "fetch_problems": problems,
        "method": (
            f"All {len(combos)} combinations of {len(with_history)} symbols were fitted in Python and "
            f"{len(candidates)} had enough overlapping history to screen; none was chosen by a model. "
            f"Every screening statistic, and therefore the ranking, is measured on the in-sample "
            f"window only, so the holdout is out of sample for the choice of pair as well as for the "
            f"parameters. Ranking is a weighted average of per-metric ranks "
            f"({', '.join(name for name, v in config.SCREEN_WEIGHTS.items() if v > 0)}). The top {k} "
            f"were then backtested on that holdout using in-sample parameters only."
        ),
    }


def _reverts(candidate: dict) -> bool:
    """Whether the spread mean-reverts fast enough to be worth trading.

    A missing half-life means the fit found no reversion at all. A very long one means the spread
    wanders: it would not return to its mean inside the maximum holding period, so every trade would
    close on the time stop rather than on the convergence the whole strategy depends on.
    """
    hl = candidate.get("half_life_days")
    return hl is not None and hl <= config.MAX_HALF_LIFE_DAYS


def _brief(candidate: dict) -> dict:
    """The short form for pairs that ranked but were not backtested."""
    return {key: candidate.get(key) for key in
            ("a", "b", "rank_score", "correlation", "half_life_days", "crossings_per_year", "beta_stability")}
