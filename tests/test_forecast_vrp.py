import numpy as np
import pandas as pd
import pytest

from volres import forecast as fc, vrp


def simulate_garch(n=6000, omega=2e-6, alpha=0.08, beta=0.9, seed=0):
    rng = np.random.default_rng(seed)
    r, s2 = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r


def test_garch_recovers_parameters():
    omega, alpha, beta = fc.garch_fit(simulate_garch())
    assert alpha == pytest.approx(0.08, abs=0.03)
    assert beta == pytest.approx(0.90, abs=0.04)


def test_garch_term_forecast_mean_reverts():
    omega, alpha, beta = 2e-6, 0.08, 0.9
    long_run_vol = np.sqrt(omega / (1 - alpha - beta) * 252)
    high = fc.garch_term_vol(np.array([10 * omega / (1 - alpha - beta)]), omega, alpha, beta, h=5000)
    assert high[0] == pytest.approx(long_run_vol, rel=0.05)


def test_forward_realized_vol_uses_only_future_days():
    idx = pd.bdate_range("2020-01-01", periods=60)
    r = pd.Series(0.0, index=idx)
    r.iloc[30] = 0.05
    rv = fc.realized_vol_forward(r, h=21)
    assert rv.iloc[30] == 0        # day 30's own return is not in its forward window
    assert rv.iloc[29] > 0 and rv.iloc[9] > 0 and rv.iloc[8] == 0


def test_har_forecast_has_no_lookahead():
    """Changing targets inside a forecast year must not change that year's forecasts."""
    idx = pd.bdate_range("2010-01-01", "2014-12-31")
    rng = np.random.default_rng(1)
    dv = pd.Series(rng.gamma(2, 5e-5, len(idx)), index=idx)
    X = fc.har_features(dv)
    target = np.sqrt(dv.rolling(21).mean().shift(-21) * 252)
    base = fc.har_forecast(target, X, 2013)
    tampered = target.copy()
    tampered[tampered.index >= "2012-12-01"] *= 5  # includes windows still open at the 2013 refit
    again = fc.har_forecast(tampered, X, 2013)
    year = base.index.year == 2013
    pd.testing.assert_series_equal(base[year], again[year])


def test_evaluate_prefers_the_true_forecast():
    rng = np.random.default_rng(2)
    true = pd.Series(rng.uniform(0.1, 0.4, 3000))
    y = true * np.sqrt(rng.chisquare(21, 3000) / 21)
    res = fc.evaluate(y, pd.DataFrame({"true": true, "noisy": true * rng.uniform(0.6, 1.4, 3000)}))
    assert res.iloc[0]["model"] == "true"


def test_short_variance_pnl():
    assert vrp.short_var_pnl(pd.Series([20.0]), pd.Series([20.0]), 0.0).iloc[0] == 0
    assert vrp.short_var_pnl(pd.Series([20.0]), pd.Series([10.0]), 0.5).iloc[0] == pytest.approx(7.5 - 0.5)
    assert vrp.short_var_pnl(pd.Series([20.0]), pd.Series([40.0]), 0.0).iloc[0] == -30  # convex losses


def test_filtered_months_earn_zero():
    pnl = pd.Series([1.0, -10.0, 1.0, 1.0])
    s = vrp.summarize(pnl, pd.Series([True, False, True, True]))
    assert s["traded"] == 3 and s["worst"] == 0 and s["mean_pnl"] == pytest.approx(0.75)
