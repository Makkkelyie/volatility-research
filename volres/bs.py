"""Black-Scholes (forward form) pricing and implied-volatility inversion."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def price(F: float, K: float, T: float, vol: float, df: float = 1.0, call: bool = True) -> float:
    """Black-76 price of a European option on forward F, discounted by `df`."""
    if T <= 0 or vol <= 0:
        intrinsic = max(F - K, 0.0) if call else max(K - F, 0.0)
        return df * intrinsic
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def vega(F: float, K: float, T: float, vol: float, df: float = 1.0) -> float:
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    return df * F * norm.pdf(d1) * np.sqrt(T)


def delta(F: float, K: float, T: float, vol: float, call: bool = True) -> float:
    """Forward delta (undiscounted)."""
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    return norm.cdf(d1) if call else norm.cdf(d1) - 1.0


def implied_vol(px: float, F: float, K: float, T: float, df: float = 1.0, call: bool = True,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """Vol that reproduces `px`; NaN if the price is outside no-arbitrage bounds."""
    intrinsic = df * (max(F - K, 0.0) if call else max(K - F, 0.0))
    upper = df * (F if call else K)
    if not (intrinsic < px < upper):
        return float("nan")
    f = lambda v: price(F, K, T, v, df, call) - px
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-10))
