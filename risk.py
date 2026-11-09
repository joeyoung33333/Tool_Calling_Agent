"""Risk statistics from daily closes. Pure Python, no dependencies.

Simple daily returns, 252 trading days a year. Prices are split-adjusted but exclude dividends, so these are price returns.
"""

import math
import statistics
from datetime import date

TRADING_DAYS = 252
FLAT = 1e-12  # a deviation below this is rounding noise (a constant series), not risk


def _pct(x: float) -> float:
    return round(x * 100, 2)


def returns(closes: list[float]) -> list[float]:
    return [b / a - 1 for a, b in zip(closes, closes[1:])]


def annual_vol(r: list[float]) -> float:
    """Volatility of daily returns, annualized, as a fraction."""
    return statistics.stdev(r) * math.sqrt(TRADING_DAYS)


def _max_drawdown(dates: list[str], closes: list[float]) -> dict:
    """Deepest fall from a peak, with its dates and the date the peak was regained (None if it has not been)."""
    peak = worst_peak = worst_trough = 0
    worst = 0.0
    for i, close in enumerate(closes):
        if close > closes[peak]:
            peak = i
        if close / closes[peak] - 1 < worst:
            worst, worst_peak, worst_trough = close / closes[peak] - 1, peak, i
    if worst == 0:
        return {"max_drawdown_pct": 0.0, "peak_date": None, "trough_date": None, "recovery_date": None}
    recovery = next((dates[i] for i in range(worst_trough, len(closes)) if closes[i] >= closes[worst_peak]), None)
    return {"max_drawdown_pct": _pct(worst), "peak_date": dates[worst_peak], "trough_date": dates[worst_trough], "recovery_date": recovery}


def _basic(dates: list[str], closes: list[float], rf: float) -> dict:
    """Figures for any series; `rf` is the daily risk-free rate."""
    r = returns(closes)
    years = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days / 365.25
    vol = annual_vol(r)
    return {
        "total_return_pct": _pct(closes[-1] / closes[0] - 1),
        "cagr_pct": _pct((closes[-1] / closes[0]) ** (1 / years) - 1),
        "volatility_pct": _pct(vol),
        "sharpe": round((statistics.fmean(r) - rf) * TRADING_DAYS / vol, 2) if vol > FLAT else None,
        **_max_drawdown(dates, closes),
    }


def summary(dates: list[str], closes: list[float], bench_closes: list[float], rf_pct: float) -> dict:
    """Risk figures for one series and a benchmark on the same dates. `rf_pct` is the annual risk-free rate in percent."""
    rf = rf_pct / 100 / TRADING_DAYS
    r, b = returns(closes), returns(bench_closes)
    out = _basic(dates, closes, rf)

    # Sortino counts only days below the risk-free rate as risk. VaR and CVaR are the 5% worst days and their average.
    shortfall = math.sqrt(statistics.fmean(min(x - rf, 0) ** 2 for x in r))
    worst_days = sorted(r)[: math.ceil(0.05 * len(r))]
    out["sortino"] = round((statistics.fmean(r) - rf) / shortfall * math.sqrt(TRADING_DAYS), 2) if shortfall > FLAT else None
    out["var_95_1d_pct"] = _pct(-worst_days[-1])
    out["cvar_95_1d_pct"] = _pct(-statistics.fmean(worst_days))

    # CAPM regression of daily excess returns on the benchmark's. A constant series has no beta.
    out.update(beta=None, alpha_annual_pct=None, correlation=None)
    if annual_vol(r) > FLAT and annual_vol(b) > FLAT:
        beta = statistics.covariance(r, b) / statistics.variance(b)
        alpha = statistics.fmean(r) - rf - beta * (statistics.fmean(b) - rf)
        out.update(beta=round(beta, 2), alpha_annual_pct=_pct(alpha * TRADING_DAYS), correlation=round(statistics.correlation(r, b), 2))
    out["benchmark_stats"] = _basic(dates, bench_closes, rf)  # the same figures for the benchmark, to compare
    return out
