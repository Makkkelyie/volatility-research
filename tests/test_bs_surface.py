import numpy as np
import pandas as pd
import pytest

from volres import bs, surface


@pytest.mark.parametrize("K", [80.0, 100.0, 125.0])
@pytest.mark.parametrize("call", [True, False])
def test_implied_vol_round_trip(K, call):
    px = bs.price(100.0, K, 0.5, 0.23, 0.98, call)
    assert bs.implied_vol(px, 100.0, K, 0.5, 0.98, call) == pytest.approx(0.23, abs=1e-8)


def test_put_call_parity():
    c, p = bs.price(100, 90, 1.0, 0.3, 0.95, True), bs.price(100, 90, 1.0, 0.3, 0.95, False)
    assert c - p == pytest.approx(0.95 * (100 - 90))


def test_price_outside_arbitrage_bounds_has_no_vol():
    assert np.isnan(bs.implied_vol(0.5, 100, 90, 0.5, 1.0, call=True))   # below intrinsic
    assert np.isnan(bs.implied_vol(101, 100, 90, 0.5, 1.0, call=True))   # above forward


def synthetic_chain(F=500.0, rate=0.04, T_days=(30, 90, 180)):
    """Chain priced from a known SVI surface, as American-style quotes would look OTM."""
    rows = []
    asof = pd.Timestamp("2026-01-02")
    for d in T_days:
        T = d / 365
        s = surface.SVI(a=0.01 * T * 4, b=0.1 * np.sqrt(T), rho=-0.6, m=0.02, sigma=0.15)
        df = np.exp(-rate * T)
        for K in np.arange(0.7 * F, 1.25 * F, 5.0):
            vol = float(s.vol(np.log(K / F), T))
            for kind in ("call", "put"):
                mid = bs.price(F, K, T, vol, df, kind == "call")
                rows.append({"expiry": (asof + pd.Timedelta(days=d)).date().isoformat(), "type": kind,
                             "strike": K, "bid": mid * 0.995, "ask": mid * 1.005, "lastPrice": mid,
                             "volume": 1, "openInterest": 1, "impliedVolatility": vol})
    return pd.DataFrame(rows).assign(spot=F * np.exp(-0.0), rate=rate, asof=asof.date().isoformat())


def test_surface_recovers_known_vols_without_arbitrage():
    chain = synthetic_chain()
    vols, stats = surface.otm_vols(chain, rate=0.04)
    assert stats["expiries"] == 3
    assert vols["F"].sub(500).abs().max() < 0.5          # forward recovered from parity
    fits = surface.fit_surface(vols)
    assert (fits["rmse_vol_pts"] < 0.05).all()
    assert (fits["min_g"] > 0).all()
    assert surface.calendar_violations(fits).empty


def test_stale_quote_is_removed():
    puts = pd.Series([1.0, 1.4, 9.0, 2.1, 2.6], index=[90, 91, 92, 93, 94])  # 92 is a stale print
    kept = surface.drop_vertical_arbitrage(puts, increasing=True)
    assert list(kept.index) == [90, 91, 93, 94]


def test_butterfly_check_flags_bad_smile():
    good = surface.SVI(0.02, 0.1, -0.5, 0.0, 0.2)
    bad = surface.SVI(0.0, 3.0, -0.99, 0.0, 0.01)  # extreme wing slope -> negative density
    k = np.linspace(-0.3, 0.3, 61)
    assert surface.butterfly_density(good, k).min() > 0
    assert surface.butterfly_density(bad, k).min() < 0
