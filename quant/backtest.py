"""Backtest one pair trade on daily bars, in-sample then on data it was not fitted on.

A screen looks at hundreds of combinations and reports the best, which is the setup that
manufactures false positives. So every choice here costs the strategy something:

- **Frozen parameters.** Beta and the spread's mean and sd come from the in-sample window only, and
  the holdout is scored with those same numbers. Refitting would measure the holdout against itself.
- **Next-open fills.** A signal from today's close is filled at tomorrow's open. Filling at the
  close you just used to decide is the commonest way a pairs backtest invents returns.
- **Costs always on.** Commission per leg per fill, plus borrow on the short leg per session held.
- **Trials correction.** `deflated_sharpe()` haircuts a Sharpe by how many pairs were searched, and
  the screen counts those trials in code so the number cannot be talked down.

Returns are price returns: split-adjusted bars with no dividends, which flatters a short leg. Said
in the output rather than silently ignored.
"""

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

import config
from quant import pairs

TRADING_DAYS = pairs.TRADING_DAYS

# What the method excludes, stated on every result including a refusal. Constant because it
# describes the method, not the data: a caller that has any result can disclose it, and the agents
# are told to read it from here rather than carry their own copy in a prompt.
LIMITATIONS = (
    "Price returns only: bars are split-adjusted but exclude dividends. Fills are at the "
    "next session's open with no slippage or market impact modelled."
)


@dataclass
class Rules:
    """Entry and exit thresholds, and costs. Adjustable per request; these are the defaults.

    Entry and exit matter most: a tight exit trades often and flatters the curve, a loose one is
    more honest about how long a divergence really takes to close.
    """

    entry_z: float = config.ENTRY_Z  # open when the spread is this many sd from its in-sample mean
    exit_z: float = config.EXIT_Z  # close when it has come back this close to the mean
    stop_z: float = config.STOP_Z  # abandon the trade: the relationship is behaving unlike the fit
    max_hold: int = config.MAX_HOLD_SESSIONS  # sessions, after which a trade is closed regardless
    cost_bps: float = config.COST_BPS  # per leg, per fill
    borrow_bps_annual: float = config.BORROW_BPS_ANNUAL  # on the short leg, per session held
    in_sample_frac: float = config.IN_SAMPLE_FRAC  # chronological split, never random

    def __post_init__(self) -> None:
        """Reject thresholds that would quietly produce a meaningless backtest.

        These arrive from the model, and with exit_z above entry_z every trade exits the session
        after it opens, reporting a busy strategy whose entire P&L is round-trip commission.
        """
        if not 0.0 <= self.exit_z < self.entry_z < self.stop_z:
            raise ValueError(
                f"entry_z={self.entry_z}, exit_z={self.exit_z} and stop_z={self.stop_z} must satisfy "
                f"0 <= exit_z < entry_z < stop_z: a trade has to open further from the mean than it "
                f"closes, and give up before the stop."
            )
        if not 0.3 <= self.in_sample_frac <= 0.9:
            raise ValueError(f"in_sample_frac={self.in_sample_frac} must be between 0.3 and 0.9 so "
                             f"both the fit and the holdout have enough sessions to mean anything.")
        if self.max_hold < 1:
            raise ValueError(f"max_hold={self.max_hold} must be at least one session.")


@dataclass
class Fit:
    """The in-sample parameters, frozen. Everything the holdout is scored with comes from here."""

    beta: float
    mean: float
    sd: float
    start: str
    end: str
    sessions: int


@dataclass
class Trade:
    """One round trip. `side` is +1 long the spread (long A, short beta*B), -1 the reverse."""

    entry_date: str
    exit_date: str
    side: int
    entry_z: float
    exit_z: float
    sessions: int
    pnl: float  # dollars, on one share of A against beta shares of B
    return_pct: float  # that P&L per dollar committed at entry, which is the comparable figure
    costs: float
    reason: str  # 'target', 'stop', 'max_hold' or 'end_of_data'


class _Open(NamedTuple):
    """The live position: where it was entered and the equity level it was opened from."""

    date: str
    index: int
    z: float
    capital: float  # gross notional of both legs at the entry fill
    cost: float  # entry commission in dollars
    spread: float  # the spread at the entry fill, which the P&L is measured against
    equity_before: float  # cumulative return before this trade, so the curve resumes from it


@dataclass
class Result:
    """What one window produced.

    `equity` is a cumulative *return*: each trade's P&L over the capital that trade committed. A
    dollar total divided by a notional re-measured each session would move the curve on days with
    no position open, silently rewriting every Sharpe and drawdown built from it.
    """

    label: str
    start: str
    end: str
    sessions: int
    trades: list[Trade] = field(default_factory=list)
    equity: list[tuple[str, float]] = field(default_factory=list)
    daily: list[float] = field(default_factory=list)  # daily return on capital


def fit(dates: list[str], a: list[float], b: list[float]) -> Fit:
    """Fit the hedge ratio and spread distribution on this window. Call on in-sample data only."""
    beta = pairs.hedge_ratio(a, b)
    s = pairs.spread(a, b, beta)
    sd = statistics.pstdev(s)
    if sd < pairs.FLAT:
        raise ValueError("the in-sample spread has no variance, so there is nothing to trade")
    return Fit(beta, statistics.fmean(s), sd, dates[0], dates[-1], len(dates))


def split(dates: list[str], rules: Rules) -> int:
    """Index where the holdout begins: a chronological split, so the holdout is strictly later."""
    return max(2, int(len(dates) * rules.in_sample_frac))


def run(
    label: str,
    dates: list[str],
    a_bars: list[dict],
    b_bars: list[dict],
    model: Fit,
    rules: Rules,
) -> Result:
    """Trade this window using `model`'s frozen parameters.

    Decisions use closes, fills the *next* session's opens, so the last bar can only close a trade.
    Capital is the gross notional at entry, so a return is per dollar committed to both legs.
    """
    result = Result(label, dates[0], dates[-1], len(dates))
    beta, mean, sd = model.beta, model.mean, model.sd
    side, entry = 0, None
    cumulative = 0.0  # a cumulative RETURN, not dollars - see Result.equity
    armed = True  # cleared by a stop or a timeout, so a broken spread is not re-entered every session

    for i, date in enumerate(dates):
        z = (a_bars[i]["close"] - beta * b_bars[i]["close"] - mean) / sd
        nxt = i + 1  # the session we could actually trade in
        can_fill = nxt < len(dates)
        equity = cumulative  # flat sessions leave the curve exactly where it was

        if side and entry:
            held = i - entry.index + 1  # sessions the position was actually open
            # Mark the open position at today's close, net of one session's borrow on the short leg.
            spread_now = a_bars[i]["close"] - beta * b_bars[i]["close"]
            gross = side * (spread_now - entry.spread)
            borrow = _borrow(beta, b_bars[i]["close"], a_bars[i]["close"], side, rules) * held
            equity = entry.equity_before + (gross - borrow - entry.cost) / entry.capital
            reason = (
                "target" if abs(z) <= rules.exit_z
                else "stop" if abs(z) >= rules.stop_z
                else "max_hold" if held >= rules.max_hold
                else "end_of_data" if not can_fill
                else ""
            )
            if reason:
                j = nxt if can_fill else i
                px_a, px_b = _fill(a_bars[j], can_fill), _fill(b_bars[j], can_fill)
                exit_spread = px_a - beta * px_b
                exit_cost = _cost(px_a, beta, px_b, rules)
                pnl = side * (exit_spread - entry.spread) - entry.cost - exit_cost - borrow
                result.trades.append(Trade(
                    entry_date=entry.date, exit_date=dates[j], side=side,
                    entry_z=round(entry.z, 2), exit_z=round(z, 2), sessions=held,
                    pnl=round(pnl, 4), return_pct=round(pnl / entry.capital * 100, 3),
                    costs=round(entry.cost + exit_cost + borrow, 4), reason=reason,
                ))
                cumulative = entry.equity_before + pnl / entry.capital
                equity = cumulative
                side, entry = 0, None
                # A stop or a timeout means the spread did not behave as the fit described, so wait
                # for it inside the exit band before trading it again. Otherwise a spread sitting at
                # |z| = 3 is re-entered every session, paying costs each time.
                armed = reason not in ("stop", "max_hold")

        else:
            if abs(z) <= rules.exit_z:
                armed = True  # the spread came home, so the relationship is worth trusting again
            # Only inside the band: at or past the stop there is nothing to do but lose the costs.
            if armed and rules.entry_z <= abs(z) < rules.stop_z and can_fill:
                # z > 0 means A is rich against B, so sell the spread.
                side = -1 if z > 0 else 1
                px_a, px_b = a_bars[nxt]["open"], b_bars[nxt]["open"]
                entry = _Open(
                    date=dates[nxt], index=nxt, z=z,
                    capital=abs(px_a) + abs(beta * px_b),
                    cost=_cost(px_a, beta, px_b, rules),
                    spread=px_a - beta * px_b,
                    equity_before=cumulative,
                )

        result.equity.append((date, round(equity, 6)))

    result.daily = [y - x for (_, x), (_, y) in zip(result.equity, result.equity[1:])]
    return result


def _fill(bar: dict, use_open: bool) -> float:
    """The execution price: the next open normally, the close only when data has run out."""
    return bar["open"] if use_open else bar["close"]


def _cost(px_a: float, beta: float, px_b: float, rules: Rules) -> float:
    """Commission for one fill of both legs, in dollars."""
    return (abs(px_a) + abs(beta * px_b)) * rules.cost_bps / 10_000


def _borrow(beta: float, px_b: float, px_a: float, side: int, rules: Rules) -> float:
    """One session's borrow on whichever leg is short, in dollars."""
    short_notional = abs(beta * px_b) if side == 1 else abs(px_a)
    return short_notional * rules.borrow_bps_annual / 10_000 / TRADING_DAYS


def metrics(result: Result) -> dict:
    """Headline figures for one window.

    Deliberately no risk-free deduction: the short leg is already charged as borrow, so carry would
    charge it twice, on every flat session. `exposure_pct` sits beside the Sharpe because one
    annualized from a mostly-zero series is not evidence.
    """
    trades, daily = result.trades, result.daily
    wins = [t for t in trades if t.pnl > 0]
    curve = [v for _, v in result.equity]
    peak, drawdown = curve[0] if curve else 0.0, 0.0
    for v in curve:
        peak = max(peak, v)
        drawdown = min(drawdown, v - peak)
    sd = statistics.pstdev(daily) if len(daily) > 2 else 0.0
    total = curve[-1] if curve else 0.0
    held = sum(t.sessions for t in trades)
    return {
        "label": result.label,
        "start": result.start,
        "end": result.end,
        "sessions": result.sessions,
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "total_return_pct": round(total * 100, 2),
        "sharpe": round(statistics.fmean(daily) / sd * math.sqrt(TRADING_DAYS), 2)
        if sd > pairs.FLAT else None,
        "exposure_pct": round(held / result.sessions * 100, 1) if result.sessions else None,
        "max_drawdown_pct": round(drawdown * 100, 2),
        "avg_hold_days": round(statistics.fmean([t.sessions for t in trades]), 1) if trades else None,
        "costs_paid": round(sum(t.costs for t in trades), 2),
        "exit_reasons": dict(Counter(t.reason for t in trades)),
    }


def sharpe_haircut(trials: int, sessions: int) -> float | None:
    """The annualized Sharpe the luckiest of `trials` worthless strategies would post over `sessions`.

    sqrt(2*ln N) is the expected maximum of N standard normals; over sqrt(sessions) it becomes the
    sampling error of a Sharpe estimate, annualized by sqrt(252). Reported beside the deflated
    figure, because a haircut of 3 says the holdout is too short to prove anything either way.
    """
    if trials < 2 or sessions < 30:
        return None
    return round(math.sqrt(2 * math.log(trials)) / math.sqrt(sessions) * math.sqrt(TRADING_DAYS), 2)


def deflated_sharpe(sharpe: float | None, trials: int, sessions: int) -> float | None:
    """`sharpe` less what the best of `trials` random strategies would have scored by luck.

    Searching N pairs and reporting the winner inflates Sharpe even when nothing works. A pair that
    survives this is worth a second look; one that does not was a coincidence found by brute force.
    """
    haircut = sharpe_haircut(trials, sessions)
    return None if sharpe is None or haircut is None else round(sharpe - haircut, 2)


def evaluate(
    a_history: list[dict],
    b_history: list[dict],
    rules: Rules | None = None,
    trials: int = 1,
) -> dict:
    """Fit in-sample, trade both windows, and report them side by side.

    The holdout row is the one that matters: same parameters, data the fit never saw. A pair strong
    in-sample and flat out-of-sample was fitted, not discovered.
    """
    rules = rules or Rules()
    dates, a_bars, b_bars = pairs.align_bars(a_history, b_history)
    if len(dates) < config.MIN_BACKTEST_SESSIONS:
        raise ValueError(f"only {len(dates)} overlapping sessions; at least "
                         f"{config.MIN_BACKTEST_SESSIONS} are needed to split a window")
    cut = split(dates, rules)
    closes = lambda bars: [bar["close"] for bar in bars]  # noqa: E731 - local alias, reads better inline
    model = fit(dates[:cut], closes(a_bars[:cut]), closes(b_bars[:cut]))

    # Refuse rather than report numbers from a spread that is not a pair trade at all.
    problem = pairs.hedge_problem(model.beta)
    if problem:
        return {
            "tradeable": False,
            "reason": problem,
            "fit": {"hedge_ratio": round(model.beta, 4),
                    "fitted_on": f"{model.start} to {model.end} ({model.sessions} sessions)"},
            "limitations": LIMITATIONS,
        }

    in_sample = run("in_sample", dates[:cut], a_bars[:cut], b_bars[:cut], model, rules)
    holdout = run("holdout", dates[cut:], a_bars[cut:], b_bars[cut:], model, rules)
    in_metrics, out_metrics = metrics(in_sample), metrics(holdout)
    out_metrics["deflated_sharpe"] = deflated_sharpe(out_metrics["sharpe"], trials, holdout.sessions)
    out_metrics["sharpe_haircut"] = sharpe_haircut(trials, holdout.sessions)
    # Below two trials there is no search to deflate for and both figures are None, so the formula
    # would read "the luckiest of 1 searched pairs" beside two nulls. analyze_pair always lands here.
    out_metrics["deflation_note"] = (
        f"deflated_sharpe = sharpe - sharpe_haircut, where the haircut is what the luckiest of "
        f"{trials} searched pairs would post by chance over {holdout.sessions} sessions. A haircut "
        f"above about 2.5 means this holdout is too short to prove anything at this search width, "
        f"so a negative deflated figure is evidence about the sample, not proof the pair is bad."
    ) if out_metrics["sharpe_haircut"] is not None else (
        "deflated_sharpe and sharpe_haircut are null because only this pair was tested, so there is "
        "no search width to deflate for. Quote sharpe and say it was not deflated."
    )

    return {
        "tradeable": True,
        "fit": {
            "hedge_ratio": round(model.beta, 4),
            "spread_mean": round(model.mean, 4),
            "spread_sd": round(model.sd, 4),
            "fitted_on": f"{model.start} to {model.end} ({model.sessions} sessions)",
            "note": "These three numbers are reused unchanged on the holdout; nothing is refitted.",
        },
        "rules": vars(rules),
        "in_sample": in_metrics,
        "holdout": out_metrics,
        "trades": [vars(t) for t in holdout.trades],
        "trials_considered": trials,
        "limitations": LIMITATIONS,
    }
