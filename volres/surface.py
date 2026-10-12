"""Implied-volatility surface from a raw option chain.

quotes -> clean -> implied forward + discount (put-call parity) -> OTM implied vols
       -> raw-SVI fit per expiry -> static-arbitrage checks (butterfly, calendar)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from volres import bs

MIN_DAYS, MAX_DAYS = 7, 400
MAX_REL_SPREAD = 0.35  # drop quotes whose bid-ask spread exceeds 35% of mid


def clean_quotes(chain: pd.DataFrame, asof: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Keep two-sided, reasonably tight quotes and add time to expiry (years)."""
    asof = pd.Timestamp(asof or chain["asof"].iloc[0])
    q = chain.copy()
    n0 = len(q)
    q["T"] = (pd.to_datetime(q["expiry"]) - asof).dt.days / 365.0
    q = q[(q["T"] * 365 >= MIN_DAYS) & (q["T"] * 365 <= MAX_DAYS)]
    n_t = len(q)
    q = q[(q["bid"] > 0) & (q["ask"] > q["bid"])]
    n_two = len(q)
    q = q.assign(mid=(q["bid"] + q["ask"]) / 2)
    q = q[(q["ask"] - q["bid"]) / q["mid"] <= MAX_REL_SPREAD]
    stats = {"raw": n0, "in_maturity_range": n_t, "two_sided": n_two, "tight_spread": len(q)}
    return q.reset_index(drop=True), stats


def implied_forward(q: pd.DataFrame, spot: float, rate: float) -> tuple[float, float] | None:
    """Forward from put-call parity, F = K + (C - P) / DF, using strikes within 3% of spot.

    SPY options are American, so deep in-the-money puts carry an early-exercise premium
    that biases a parity regression across all strikes (it implies DF > 1). Near the money
    that premium is negligible, so DF is taken from the T-bill rate and F is the median over
    near-the-money pairs. The forward absorbs dividends, so no dividend assumption is needed.
    """
    calls = q[q["type"] == "call"].set_index("strike")["mid"]
    puts = q[q["type"] == "put"].set_index("strike")["mid"]
    both = calls.index.intersection(puts.index)
    both = both[(both > 0.97 * spot) & (both < 1.03 * spot)]
    if len(both) < 3:
        return None
    T = float(q["T"].iloc[0])
    df = float(np.exp(-rate * T))
    F = float(np.median(both.to_numpy(float) + (calls[both] - puts[both]).to_numpy() / df))
    return F, df


def drop_vertical_arbitrage(prices: pd.Series, increasing: bool) -> pd.Series:
    """Remove quotes that break price monotonicity in strike (puts must get dearer as the
    strike rises, calls cheaper). Repeatedly drops the quote involved in the largest
    violation, so one stale print doesn't take its honest neighbours with it."""
    p = prices.sort_index()
    while len(p) > 2:
        step = p.diff().iloc[1:] if increasing else -p.diff().iloc[1:]
        if (step >= 0).all():
            break
        bad = step.idxmin()  # higher strike of the worst pair
        i = p.index.get_loc(bad)
        # drop whichever of the pair is further from the median of its neighbours
        lo, hi = p.index[i - 1], bad
        neigh = p.iloc[max(i - 3, 0): i + 3].median()
        p = p.drop(lo if abs(p[lo] - neigh) > abs(p[hi] - neigh) else hi)
    return p


def otm_vols(chain: pd.DataFrame, rate: float) -> tuple[pd.DataFrame, dict]:
    """Implied vols of out-of-the-money options, one row per (expiry, strike).

    Only OTM options are used: they are the liquid side of each strike and their
    early-exercise premium is small, so the European Black-76 inversion is a fair proxy."""
    q, stats = clean_quotes(chain)
    spot = float(chain["spot"].iloc[0])
    rows = []
    for expiry, g in q.groupby("expiry"):
        fwd = implied_forward(g, spot, rate)
        if fwd is None:
            continue
        F, df = fwd
        T = float(g["T"].iloc[0])
        otm = g[((g["type"] == "put") & (g["strike"] < F)) | ((g["type"] == "call") & (g["strike"] >= F))]
        keep = []
        for kind, increasing in (("put", True), ("call", False)):
            side = otm[otm["type"] == kind].set_index("strike")["mid"]
            keep += [(kind, k) for k in drop_vertical_arbitrage(side, increasing).index]
        n_before = len(otm)
        otm = otm[[(t, k) in set(keep) for t, k in zip(otm["type"], otm["strike"])]]
        stats["vertical_arb_dropped"] = stats.get("vertical_arb_dropped", 0) + n_before - len(otm)
        for r in otm.itertuples():
            iv = bs.implied_vol(r.mid, F, r.strike, T, df, call=r.type == "call")
            if np.isfinite(iv):
                rows.append({"expiry": expiry, "T": T, "F": F, "df": df, "strike": r.strike,
                             "type": r.type, "mid": r.mid, "iv": iv, "yahoo_iv": r.impliedVolatility,
                             "k": np.log(r.strike / F), "w": iv * iv * T})
    out = pd.DataFrame(rows)
    stats["otm_vols"] = len(out)
    stats["expiries"] = out["expiry"].nunique() if len(out) else 0
    return out, stats


# ---------------------------------------------------------------- SVI


@dataclass(frozen=True)
class SVI:
    """Raw SVI total variance: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def w(self, k):
        k = np.asarray(k, float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def dw(self, k):
        x = np.asarray(k, float) - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.sigma ** 2))

    def d2w(self, k):
        x = np.asarray(k, float) - self.m
        return self.b * self.sigma ** 2 / (x * x + self.sigma ** 2) ** 1.5

    def vol(self, k, T):
        return np.sqrt(np.maximum(self.w(k), 1e-12) / T)


def fit_svi(k: np.ndarray, w: np.ndarray, weights: np.ndarray | None = None) -> SVI:
    """Least-squares SVI fit with parameter bounds, best of several starting points."""
    k, w = np.asarray(k, float), np.asarray(w, float)
    wt = np.ones_like(w) if weights is None else np.asarray(weights, float)
    w_atm = float(np.interp(0.0, np.sort(k), w[np.argsort(k)]))

    def resid(p):
        s = SVI(*p)
        # soft penalty keeps the minimum total variance non-negative
        floor = p[0] + p[1] * p[4] * np.sqrt(1 - p[2] ** 2)
        return np.append(wt * (s.w(k) - w), 10 * min(floor, 0.0))

    lo = [-1.0, 1e-6, -0.999, -1.0, 1e-4]
    hi = [1.0, 5.0, 0.999, 1.0, 2.0]
    best = None
    for rho0 in (-0.7, -0.3, 0.0):
        for sig0 in (0.05, 0.2):
            p0 = [max(w_atm * 0.5, 1e-4), 0.1, rho0, 0.0, sig0]
            # soft-L1 loss: a few bad quotes shouldn't bend the whole smile
            res = least_squares(resid, p0, bounds=(lo, hi), method="trf", loss="soft_l1", f_scale=1e-4)
            if best is None or res.cost < best.cost:
                best = res
    return SVI(*best.x)


def butterfly_density(s: SVI, k: np.ndarray) -> np.ndarray:
    """Gatheral's g(k); g < 0 anywhere means the smile implies a negative risk-neutral density."""
    w, w1, w2 = s.w(k), s.dw(k), s.d2w(k)
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


def fit_surface(vols: pd.DataFrame) -> pd.DataFrame:
    """Fit SVI to every expiry; report fit error and arbitrage diagnostics."""
    rows = []
    for expiry, g in vols.groupby("expiry"):
        if len(g) < 8:
            continue
        T = float(g["T"].iloc[0])
        # focus the fit on the liquid part of the smile
        atm_sd = float(np.sqrt(np.interp(0.0, np.sort(g["k"]), g["w"].to_numpy()[np.argsort(g["k"])])))
        g = g[g["k"].abs() <= max(4 * atm_sd, 0.08)]
        if len(g) < 8:
            continue
        s = fit_svi(g["k"].to_numpy(), g["w"].to_numpy())
        fit_vol = s.vol(g["k"].to_numpy(), T)
        h = 1e-3
        rows.append({
            "expiry": expiry, "T": T, "n_quotes": len(g), "svi": s,
            "atm_vol": float(s.vol(0.0, T)),
            # skew: d(implied vol)/d(log-moneyness) at the money, per 10% moneyness
            "atm_skew_10pct": float((s.vol(h, T) - s.vol(-h, T)) / (2 * h) * 0.10),
            "rmse_vol_pts": float(np.sqrt(np.mean((fit_vol - g["iv"]) ** 2)) * 100),
            # butterfly check only where there are quotes; SVI wings beyond that are extrapolation
            "min_g": float(butterfly_density(s, np.linspace(g["k"].min(), g["k"].max(), 200)).min()),
            "k_min": float(g["k"].min()), "k_max": float(g["k"].max()),
        })
    out = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
    return out


def calendar_violations(fits: pd.DataFrame, n_grid: int = 25) -> pd.DataFrame:
    """Adjacent expiries whose total variance *decreases* with maturity somewhere on the
    log-moneyness range both expiries actually quote (an arbitrage: buy the long-dated, sell
    the short-dated calendar spread)."""
    rows = []
    for (_, a), (_, b) in zip(fits.iloc[:-1].iterrows(), fits.iloc[1:].iterrows()):
        lo, hi = max(a["k_min"], b["k_min"]), min(a["k_max"], b["k_max"])
        k = np.linspace(lo, hi, n_grid)
        gap = b["svi"].w(k) - a["svi"].w(k)
        if (gap < -1e-6).any():
            worst = int(np.argmin(gap))
            rows.append({"short": a["expiry"], "long": b["expiry"], "share_of_grid": float((gap < -1e-6).mean()),
                         "worst_k": float(k[worst]),
                         "vol_gap_pts": float((np.sqrt(b["svi"].w(k[worst]) / b["T"])
                                               - np.sqrt(a["svi"].w(k[worst]) / a["T"])) * 100)})
    return pd.DataFrame(rows)
