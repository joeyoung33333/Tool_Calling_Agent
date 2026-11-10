"""Pair statistics from two price series. Pure Python, no dependencies.

The arithmetic is all in this module because the project rests on the model *not* being the thing
that decides which pairs are good. An agent chooses the universe and interprets what comes back;
these functions decide what survives.

Vocabulary, used consistently below:

    hedge ratio (beta)  how many units of B to short per unit of A, from an OLS fit of A on B
    spread              A - beta*B, the thing that is supposed to be mean-reverting
    z-score             how many standard deviations the spread sits from its mean
    half-life           sessions for a divergence to decay halfway back, from an AR(1) fit
    crossings           how often the spread actually returns through its mean

A pair is interesting when the spread mean-reverts *and* the hedge ratio is stable. Correlation
alone is not enough: two names can be 0.95 correlated while their spread trends away forever.
"""

import math
import statistics

TRADING_DAYS = 252
FLAT = 1e-12  # a deviation below this is rounding noise, not variation

# A tradeable hedge ratio must be positive, or the spread holds both legs long and carries full
# market exposure. The bounds also reject ratios so lopsided that one leg barely participates.
MIN_HEDGE_RATIO = 0.05
MAX_HEDGE_RATIO = 20.0


def align_bars(a: list[dict], b: list[dict]) -> tuple[list[str], list[dict], list[dict]]:
    """The two series' full bars on their common dates, oldest first: (dates, a_bars, b_bars).

    Sessions present for only one leg are dropped, not filled: a holiday one venue observes and the
    other does not would invent a return. Whole bars, because a backtest fills at the next open.
    """
    by_date = {bar["date"]: bar for bar in b}
    both = [(bar["date"], bar, by_date[bar["date"]]) for bar in a if bar["date"] in by_date]
    both.sort(key=lambda row: row[0])
    return [r[0] for r in both], [r[1] for r in both], [r[2] for r in both]


def align(a: list[dict], b: list[dict]) -> tuple[list[str], list[float], list[float]]:
    """The two series' closes on their common dates, oldest first: (dates, a_closes, b_closes)."""
    dates, a_bars, b_bars = align_bars(a, b)
    return dates, [bar["close"] for bar in a_bars], [bar["close"] for bar in b_bars]


def returns(closes: list[float]) -> list[float]:
    """Simple daily returns."""
    return [y / x - 1 for x, y in zip(closes, closes[1:])]


def annual_vol(r: list[float]) -> float:
    """Annualized standard deviation of a daily return series, in percent."""
    return statistics.pstdev(r) * math.sqrt(TRADING_DAYS) * 100 if len(r) > 1 else 0.0


def correlation(a: list[float], b: list[float]) -> float | None:
    """Pearson correlation of two return series. None if either is constant."""
    if len(a) < 3 or statistics.pstdev(a) < FLAT or statistics.pstdev(b) < FLAT:
        return None
    return statistics.correlation(a, b)


def hedge_ratio(a: list[float], b: list[float]) -> float:
    """OLS slope of `a` on `b`: how much of B hedges one unit of A.

    Fitted on prices, not returns, because the spread we trade is a price difference.
    """
    if len(a) < 3:
        raise ValueError("a hedge ratio needs at least 3 observations")
    var = statistics.variance(b)
    if var < FLAT:
        raise ValueError("the second leg is constant, so it cannot hedge anything")
    return statistics.covariance(a, b) / var


def hedge_problem(beta: float) -> str | None:
    """Why `beta` cannot be traded as a pair, or None if it can.

    Price-level OLS returns a negative slope whenever two price levels drift apart, even for names
    whose returns correlate well. That is the regression reporting "not cointegrated here", and it
    must never reach a backtest, where it becomes a long-both-legs position called market-neutral.
    """
    if beta <= 0:
        return (f"the hedge ratio is {beta:.4f}, which is negative: the two price levels drifted "
                "apart over this window, so the spread would be long both legs rather than market "
                "neutral. These names are not cointegrated here.")
    if beta < MIN_HEDGE_RATIO:
        return f"the hedge ratio is {beta:.4f}, so the second leg barely hedges anything"
    if beta > MAX_HEDGE_RATIO:
        return f"the hedge ratio is {beta:.1f}, which needs an implausibly large second leg"
    return None


def spread(a: list[float], b: list[float], beta: float) -> list[float]:
    """A - beta*B, the series that is supposed to mean-revert."""
    return [x - beta * y for x, y in zip(a, b)]


def zscores(values: list[float], mean: float, sd: float) -> list[float]:
    """`values` expressed in standard deviations from a *given* mean and sd.

    Arguments rather than computed here: a holdout must score against the in-sample fit, and
    recomputing them on the holdout would leak the answer.
    """
    if sd < FLAT:
        raise ValueError("a zero-variance spread has no z-score")
    return [(v - mean) / sd for v in values]


def half_life(values: list[float]) -> float | None:
    """Sessions for a deviation to decay halfway back, from an AR(1) fit of the spread.

    Regress the change on the level: a negative slope means reversion and its rate gives the
    half-life. None when the fit is non-reverting, which says the spread wanders rather than returns.
    """
    if len(values) < 10:
        return None
    level, change = values[:-1], [y - x for x, y in zip(values, values[1:])]
    if statistics.variance(level) < FLAT:
        return None
    slope = statistics.covariance(change, level) / statistics.variance(level)
    if slope >= 0:
        return None  # not mean-reverting on this window
    return math.log(2) / -math.log1p(slope) if slope > -1 else 0.5


def crossings(values: list[float], mean: float) -> int:
    """How many times the series crosses `mean`. Frequent crossings are the trade actually working."""
    sides = [v > mean for v in values]
    return sum(1 for x, y in zip(sides, sides[1:]) if x != y)


def rolling_beta(a: list[float], b: list[float], window: int = 126) -> list[float]:
    """Hedge ratio recomputed over each `window` of sessions (126 is about six months).

    A pair whose beta drifts is not one relationship but a different one in each regime, and
    backtesting it on a single number is self-deception.
    """
    if len(a) < window + 1:
        return []
    out = []
    for i in range(window, len(a) + 1):
        try:
            out.append(hedge_ratio(a[i - window:i], b[i - window:i]))
        except ValueError:
            continue
    return out


def beta_stability(betas: list[float]) -> float | None:
    """Spread of the rolling hedge ratio relative to its level: 0 is rock steady, higher is drifting.

    Scaled by |mean| so it compares across pairs with very different ratios.
    """
    if len(betas) < 3:
        return None
    mean = statistics.fmean(betas)
    return None if abs(mean) < 1e-6 else statistics.stdev(betas) / abs(mean)


def summary(dates: list[str], a: list[float], b: list[float]) -> dict:
    """Every statistic for one pair over the window it is given.

    This is the screening view. A backtest instead fits on part of the window and scores on the
    rest (see backtest.py), which is the only way to tell a real relationship from a fitted one.
    """
    ra, rb = returns(a), returns(b)
    beta = hedge_ratio(a, b)
    s = spread(a, b, beta)
    mean, sd = statistics.fmean(s), statistics.pstdev(s)
    betas = rolling_beta(a, b)
    hl = half_life(s)
    problem = hedge_problem(beta)
    crossed = crossings(s, mean)
    return {
        "sessions": len(dates),
        "tradeable": problem is None,
        "not_tradeable_because": problem,
        "start": dates[0],
        "end": dates[-1],
        "correlation": _round(correlation(ra, rb), 3),
        "hedge_ratio": round(beta, 4),
        "current_z": _round((s[-1] - mean) / sd if sd >= FLAT else None, 2),
        "half_life_days": _round(hl, 1),
        # Per interval between sessions, of which there are one fewer than there are sessions.
        "crossings_per_year": _round(crossed / max(len(dates) - 1, 1) * TRADING_DAYS, 1),
        "beta_stability": _round(beta_stability(betas), 3),
    }


def episodes(dates: list[str], z: list[float], threshold: float = 2.0, min_days: int = 10) -> list[dict]:
    """Runs where |z| stayed above `threshold` for at least `min_days` consecutive sessions.

    The divergences worth explaining: a spread two sigma wide for a fortnight is either a
    dislocation that closed or a relationship that broke, and only news or a filing over exactly
    that window tells you which. Each run carries its dates, so the research cannot drift.
    """
    out, start = [], None
    for i, value in enumerate(z + [0.0]):  # the sentinel closes a run still open at the end
        wide = abs(value) > threshold and i < len(z)
        if wide and start is None:
            start = i
        elif not wide and start is not None:
            if i - start >= min_days:
                run = z[start:i]
                peak = max(range(len(run)), key=lambda k: abs(run[k]))
                out.append({
                    "start": dates[start],
                    "end": dates[i - 1],
                    "sessions": i - start,
                    "direction": "a_rich" if statistics.fmean(run) > 0 else "a_cheap",
                    "peak_z": round(run[peak], 2),
                    "peak_date": dates[start + peak],
                    "resolved": abs(z[i]) < 1.0 if i < len(z) else False,
                })
            start = None
    return out


def _round(value: float | None, digits: int) -> float | None:
    """round() that passes None through, so a missing statistic stays missing."""
    return None if value is None else round(value, digits)
