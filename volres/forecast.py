"""Forecasting next-month (21 trading day) realized volatility of the S&P 500.

Every forecast dated t uses information up to and including the close of day t. Fitted
models (GARCH, HAR) are re-estimated once a year on an expanding window, and a training
row is only used once its 21-day target window has fully elapsed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

H = 21  # horizon in trading days
ANN = 252


def realized_vol_forward(r: pd.Series, h: int = H) -> pd.Series:
    """Annualized close-to-close vol over days t+1..t+h (the forecast target, dated t)."""
    fwd_sum = (r ** 2).rolling(h).sum().shift(-h)
    return np.sqrt(fwd_sum * ANN / h)


def parkinson_var(high: pd.Series, low: pd.Series) -> pd.Series:
    """Daily variance from the high-low range; robust to the fake opens in old index data."""
    return np.log(high / low) ** 2 / (4 * np.log(2))


# ---------------------------------------------------------------- simple benchmarks


def historical(r: pd.Series, h: int = H) -> pd.Series:
    return np.sqrt((r ** 2).rolling(h).sum() * ANN / h)


def ewma(r: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics: sigma^2_{t+1} = lam * sigma^2_t + (1 - lam) * r_t^2 (flat term structure)."""
    var = (r ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * ANN)


# ---------------------------------------------------------------- GARCH(1,1)


def garch_filter(r: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Conditional variances sigma^2_{t+1 | t} for each t (one step ahead)."""
    var = np.empty(len(r))
    s2 = np.var(r)
    for t, x in enumerate(r):
        s2 = omega + alpha * x * x + beta * s2
        var[t] = s2
    return var


def garch_fit(r: np.ndarray) -> tuple[float, float, float]:
    """Gaussian quasi-MLE of (omega, alpha, beta), with alpha + beta < 1."""
    r = r - r.mean()
    v0 = np.var(r)

    def nll(p):
        omega, alpha, beta = p
        if alpha + beta >= 0.9999:
            return 1e10
        var = np.empty(len(r))
        var[0] = v0
        s2 = v0
        for t in range(1, len(r)):
            s2 = omega + alpha * r[t - 1] ** 2 + beta * s2
            var[t] = s2
        return 0.5 * np.sum(np.log(var) + r ** 2 / var)

    res = minimize(nll, [v0 * 0.05, 0.08, 0.9], method="L-BFGS-B",
                   bounds=[(1e-12, v0), (1e-4, 0.5), (0.3, 0.9999)])
    return tuple(res.x)


def garch_term_vol(var_next: np.ndarray, omega: float, alpha: float, beta: float, h: int = H) -> np.ndarray:
    """Annualized average vol over the next h days from the one-step variance forecast."""
    persistence = alpha + beta
    long_run = omega / (1 - persistence)
    # sum of expected daily variances over steps 1..h: mean-reverts to the long-run level
    total = h * long_run + (var_next - long_run) * (persistence ** np.arange(h)).sum()
    return np.sqrt(total * ANN / h)


def garch_forecast(r: pd.Series, start_year: int) -> tuple[pd.Series, pd.DataFrame]:
    """Refit yearly on all returns before Jan 1 of each year; forecast through that year."""
    r = r.dropna()
    out = pd.Series(np.nan, index=r.index)
    params = []
    for year in range(start_year, r.index.year.max() + 1):
        train = r[r.index < f"{year}-01-01"].to_numpy()
        omega, alpha, beta = garch_fit(train)
        params.append({"year": year, "omega": omega, "alpha": alpha, "beta": beta})
        upto = r[r.index < f"{year + 1}-01-01"]
        var_next = garch_filter((upto - train.mean()).to_numpy(), omega, alpha, beta)
        mask = upto.index.year == year
        out[upto.index[mask]] = garch_term_vol(var_next[mask], omega, alpha, beta)
    return out, pd.DataFrame(params)


# ---------------------------------------------------------------- HAR


def har_features(daily_var: pd.Series) -> pd.DataFrame:
    """Daily, weekly (5d) and monthly (22d) averages of daily realized variance."""
    return pd.DataFrame({
        "rv_d": daily_var,
        "rv_w": daily_var.rolling(5).mean(),
        "rv_m": daily_var.rolling(22).mean(),
    })


def har_forecast(target_vol: pd.Series, X: pd.DataFrame, start_year: int) -> pd.Series:
    """OLS of next-month variance on HAR features (+ any extra columns, e.g. VIX^2).

    Refit yearly; a row dated t is in the training set only if t + H < refit date, so no
    part of its target window overlaps the forecast period.
    """
    y = (target_vol ** 2 / ANN)  # daily-variance units, like the regressors
    data = X.join(y.rename("y")).dropna(subset=list(X.columns))
    out = pd.Series(np.nan, index=data.index)
    dates = data.index
    for year in range(start_year, dates.year.max() + 1):
        cutoff = pd.Timestamp(f"{year}-01-01")
        pos = dates.searchsorted(cutoff)
        train = data.iloc[: max(pos - H, 0)].dropna()
        A = np.column_stack([np.ones(len(train)), train[X.columns].to_numpy()])
        coef, *_ = np.linalg.lstsq(A, train["y"].to_numpy(), rcond=None)
        test = data[data.index.year == year]
        pred = coef[0] + test[X.columns].to_numpy() @ coef[1:]
        out[test.index] = np.sqrt(np.clip(pred, 1e-8, None) * ANN)
    return out


# ---------------------------------------------------------------- evaluation


def evaluate(target: pd.Series, forecasts: pd.DataFrame) -> pd.DataFrame:
    """RMSE (vol points), QLIKE (variance; robust to noisy targets), Mincer-Zarnowitz fit."""
    rows = []
    for name in forecasts:
        d = pd.concat([target.rename("y"), forecasts[name].rename("f")], axis=1).dropna()
        y2, f2 = d["y"] ** 2, d["f"] ** 2
        ratio = y2 / f2
        slope, intercept = np.polyfit(d["f"], d["y"], 1)
        r2 = np.corrcoef(d["f"], d["y"])[0, 1] ** 2
        rows.append({
            "model": name, "n": len(d),
            "rmse_vol_pts": float(np.sqrt(np.mean((d["y"] - d["f"]) ** 2)) * 100),
            "qlike": float(np.mean(ratio - np.log(ratio) - 1)),
            "mean_bias_vol_pts": float((d["f"] - d["y"]).mean() * 100),
            "mz_slope": float(slope), "mz_r2": float(r2),
        })
    return pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)
