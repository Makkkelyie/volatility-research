"""Variance risk premium: does selling 1-month S&P 500 variance at the VIX level pay, and
can simple, pre-specified filters avoid the worst months?

Trade: every 21 trading days, sell a 1-month variance swap struck at K = VIX (vol points),
sized at 1 unit of vega notional. P&L at expiry = (K^2 - RV^2) / (2K) - cost, where RV is the
realized close-to-close vol over the next 21 days. Non-overlapping trades, so each P&L is
one independent-ish observation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

H = 21


def short_var_pnl(vix: pd.Series, rv_fwd_pts: pd.Series, cost_pts: float) -> pd.Series:
    k = vix
    return (k ** 2 - rv_fwd_pts ** 2) / (2 * k) - cost_pts


def trade_dates(index: pd.DatetimeIndex, offset: int = 0) -> pd.DatetimeIndex:
    return index[offset::H]


def summarize(pnl: pd.Series, active: pd.Series | None = None) -> dict:
    """Stats over all rebalance dates; inactive (filtered-out) months earn 0."""
    p = pnl if active is None else pnl.where(active, 0.0)
    traded = p[p != 0] if active is not None else p
    cum = p.cumsum()
    years = len(p) * H / 252
    sd = p.std()
    return {
        "months": len(p), "traded": int(len(traded)),
        "mean_pnl": float(p.mean()),
        "ann_sharpe": float(p.mean() / sd * np.sqrt(252 / H)) if sd > 0 else float("nan"),
        "t_stat": float(p.mean() / sd * np.sqrt(len(p))) if sd > 0 else float("nan"),
        "hit_rate": float((traded > 0).mean()),
        "skew": float(traded.skew()),
        "worst": float(p.min()),
        "max_drawdown": float((cum - cum.cummax()).min()),
        "pnl_per_year": float(p.sum() / years),
    }


def strategy_table(df: pd.DataFrame, cost_pts: float, offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`df` needs: vix, vix3m, rv_fwd (vol pts), har (HAR vol forecast, vol pts)."""
    d = df.loc[trade_dates(df.index, offset)].dropna(subset=["vix", "rv_fwd"])
    d = d.assign(pnl=short_var_pnl(d["vix"], d["rv_fwd"], cost_pts))
    rules = {
        "always short": None,
        "short if VIX < VIX3M (contango)": d["vix"] < d["vix3m"],
        "short if VIX - HAR forecast > 2 pts": (d["vix"] - d["har"]) > 2,
        "both filters": (d["vix"] < d["vix3m"]) & ((d["vix"] - d["har"]) > 2),
    }
    rows = []
    for name, active in rules.items():
        # compare every rule on the same months: those where all its inputs exist
        sub = d.dropna(subset=["vix3m", "har"])
        act = None if active is None else active.loc[sub.index]
        rows.append({"rule": name, **summarize(sub["pnl"], act)})
    return pd.DataFrame(rows), d


def by_period(d: pd.DataFrame, rule: pd.Series | None, cuts: list[str]) -> pd.DataFrame:
    rows = []
    edges = [d.index.min()] + [pd.Timestamp(c) for c in cuts] + [d.index.max() + pd.Timedelta(days=1)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        part = d[(d.index >= lo) & (d.index < hi)]
        act = None if rule is None else rule.loc[part.index]
        rows.append({"period": f"{lo:%Y}-{(hi - pd.Timedelta(days=1)):%Y}", **summarize(part["pnl"], act)})
    return pd.DataFrame(rows)
