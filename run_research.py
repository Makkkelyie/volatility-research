"""Volatility research: SPY implied-vol surface, S&P 500 realized-vol forecasting, and a
validated test of the variance risk premium.

    python run_research.py              # uses cached data in data/
    python run_research.py --refresh    # re-download history and snapshot today's SPY chain
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volres import data, forecast as fc, surface, vrp

OUT = Path("outputs")
COST = 0.5  # vol points per trade, a conservative variance-swap bid/ask haircut


def md(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- part 1: surface


def surface_section(chain: pd.DataFrame) -> str:
    rate = float(chain["rate"].iloc[0])
    vols, stats = surface.otm_vols(chain, rate)
    fits = surface.fit_surface(vols)
    cal = surface.calendar_violations(fits)
    asof, spot = chain["asof"].iloc[0], float(chain["spot"].iloc[0])

    # smiles: market vs SVI for a spread of maturities
    pick = fits.iloc[np.unique(np.linspace(0, len(fits) - 1, 6).round().astype(int))]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
    for ax, (_, f) in zip(axes.ravel(), pick.iterrows()):
        g = vols[vols["expiry"] == f["expiry"]]
        g = g[(g["k"] >= f["k_min"]) & (g["k"] <= f["k_max"])]
        k = np.linspace(f["k_min"], f["k_max"], 200)
        ax.scatter(g["k"], g["iv"] * 100, s=8, alpha=0.6, label="market (OTM mid)")
        ax.plot(k, f["svi"].vol(k, f["T"]) * 100, "r", lw=1.5, label="SVI fit")
        ax.set_title(f"{f['expiry']}  ({f['T'] * 365:.0f}d)  RMSE {f['rmse_vol_pts']:.2f} vol pts", fontsize=9)
        ax.set_xlabel("log-moneyness ln(K/F)")
        ax.set_ylabel("implied vol (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"SPY implied-volatility smiles, {asof} (spot {spot:.2f})")
    fig.tight_layout()
    fig.savefig(OUT / "smiles.png", dpi=120)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(fits["T"] * 365, fits["atm_vol"] * 100, "o-")
    a1.set(title="ATM implied vol term structure", xlabel="days to expiry", ylabel="ATM vol (%)")
    a2.plot(fits["T"] * 365, -fits["atm_skew_10pct"] * 100, "o-", color="C3")
    a2.set(title="ATM skew: vol pts gained per 10% move down in strike", xlabel="days to expiry", ylabel="vol pts")
    fig.tight_layout()
    fig.savefig(OUT / "term_structure.png", dpi=120)
    plt.close(fig)

    write_surface_html(fits, asof)

    table = fits[["expiry", "n_quotes", "atm_vol", "atm_skew_10pct", "rmse_vol_pts", "min_g"]].copy()
    table["atm_vol"] *= 100
    table["atm_skew_10pct"] *= 100
    table = table.rename(columns={"atm_vol": "atm_vol_%", "atm_skew_10pct": "skew_per_10%_pts",
                                  "min_g": "min_butterfly_g"})
    yahoo_gap = float(np.median(np.abs(vols["iv"] - vols["yahoo_iv"])) * 100)
    return f"""## 1. SPY implied-volatility surface ({asof})

Quote funnel: {stats['raw']:,} raw quotes -> {stats['in_maturity_range']:,} with 7-400 days to expiry ->
{stats['two_sided']:,} two-sided -> {stats['tight_spread']:,} with spread <= 35% of mid ->
{stats['vertical_arb_dropped']} dropped for breaking strike monotonicity (stale prints) ->
**{stats['otm_vols']:,} OTM implied vols across {stats['expiries']} expiries**.

- Forward per expiry from put-call parity on strikes within 3% of spot, discounted at the 13-week
  T-bill ({rate:.2%}); the forward absorbs dividends.
- Own Black-76 inversion (Brent); median gap to Yahoo's quoted IV: {yahoo_gap:.2f} vol pts.
- Raw-SVI fit per expiry (soft-L1 loss). Butterfly check (Gatheral g(k) >= 0 on the quoted range):
  min g = {table['min_butterfly_g'].min():.3f}. Calendar check (total variance non-decreasing in
  maturity on overlapping quoted range): **{len(cal)} violating expiry pairs**.

{md(table)}

![smiles](smiles.png)
![term structure](term_structure.png)

Interactive 3-D surface: `outputs/vol_surface.html`.
"""


def write_surface_html(fits: pd.DataFrame, asof: str) -> None:
    import plotly.graph_objects as go

    k = np.linspace(-0.25, 0.10, 60)
    z = np.array([s.vol(k, t) * 100 for s, t in zip(fits["svi"], fits["T"])])
    # blank out extrapolated regions so the plot only shows what the market quotes
    for i, (_, f) in enumerate(fits.iterrows()):
        z[i, (k < f["k_min"]) | (k > f["k_max"])] = np.nan
    fig = go.Figure(go.Surface(x=k, y=fits["T"] * 365, z=z, colorscale="Viridis",
                               colorbar=dict(title="IV %")))
    fig.update_layout(title=f"SPY SVI implied-vol surface, {asof}",
                      scene=dict(xaxis_title="ln(K/F)", yaxis_title="days to expiry", zaxis_title="implied vol (%)"))
    fig.write_html(OUT / "vol_surface.html", include_plotlyjs="cdn")


# ---------------------------------------------------------------- part 2: forecasting


def forecast_section(h: pd.DataFrame) -> tuple[str, pd.Series, pd.Series]:
    r = data.log_returns(h["close"])
    target = fc.realized_vol_forward(r)
    X = fc.har_features(fc.parkinson_var(h["high"], h["low"]))
    X_vix = X.assign(vix2=(h["vix"] / 100) ** 2 / fc.ANN)
    garch, params = fc.garch_forecast(r, 2000)
    har_long = fc.har_forecast(target, X, 1995)  # used by the VRP filter from 1995
    f = pd.DataFrame({
        "historical 21d": fc.historical(r), "EWMA (0.94)": fc.ewma(r), "GARCH(1,1)": garch,
        "HAR": har_long, "HAR + VIX": fc.har_forecast(target, X_vix, 2000), "VIX (implied)": h["vix"] / 100,
    })
    f = f[f.index >= "2000-01-01"]
    table = fc.evaluate(target, f)

    recent = f[f.index >= "2018-01-01"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(target.loc[recent.index] * 100, color="k", lw=1, label="realized next 21d")
    for col, c in [("HAR + VIX", "C0"), ("GARCH(1,1)", "C2"), ("VIX (implied)", "C3")]:
        ax.plot(recent[col] * 100, lw=0.9, color=c, label=col, alpha=0.85)
    ax.set(title="1-month S&P 500 volatility: forecasts vs realized (out-of-sample)", ylabel="annualized vol (%)")
    ax.set_ylim(0, 90)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "forecasts.png", dpi=120)
    plt.close(fig)

    p = params.iloc[-1]
    text = f"""## 2. Forecasting next-month realized volatility (S&P 500, out-of-sample 2000-{f.index.max():%Y})

Target: annualized close-to-close vol over the next 21 trading days. GARCH and HAR are re-fitted
every January on an expanding window; HAR training rows are only used once their 21-day target
window has closed, so no forecast sees its own future. QLIKE is the main loss: it ranks variance
forecasts consistently even when the realized target is noisy.

{md(table.rename(columns={"mean_bias_vol_pts": "bias (f - y) pts"}), "{:.3f}")}

Latest GARCH(1,1): alpha = {p['alpha']:.3f}, beta = {p['beta']:.3f} (persistence {p['alpha'] + p['beta']:.3f},
half-life {np.log(0.5) / np.log(p['alpha'] + p['beta']):.0f} days).

![forecasts](forecasts.png)
"""
    return text, target, har_long


# ---------------------------------------------------------------- part 3: VRP


def vrp_section(h: pd.DataFrame, target: pd.Series, har: pd.Series) -> str:
    df = pd.DataFrame({"vix": h["vix"], "vix3m": h["vix3m"], "rv_fwd": target * 100, "har": har * 100})
    table, d = vrp.strategy_table(df, COST)
    sub = d.dropna(subset=["vix3m", "har"])

    # robustness: the result shouldn't hinge on which day of the month we roll
    rob = []
    for off in range(vrp.H):
        t, _ = vrp.strategy_table(df, COST, off)
        rob.append(t.set_index("rule")["ann_sharpe"])
    rob = pd.DataFrame(rob)
    robust = pd.DataFrame({"rule": rob.columns, "min_sharpe": rob.min().values,
                           "median_sharpe": rob.median().values, "max_sharpe": rob.max().values,
                           "share_of_offsets_better_than_always_short":
                               [(rob[c] > rob["always short"]).mean() for c in rob.columns]})

    contango = sub["vix"] < sub["vix3m"]
    periods_all = vrp.by_period(sub, None, ["2013-01-01", "2020-01-01"])
    periods_con = vrp.by_period(sub, contango, ["2013-01-01", "2020-01-01"])
    full = df.loc[vrp.trade_dates(df.index)].dropna(subset=["vix", "rv_fwd"])
    full = full.assign(pnl=vrp.short_var_pnl(full["vix"], full["rv_fwd"], COST))
    long_run = vrp.summarize(full["pnl"])
    worst = sub.nsmallest(5, "pnl")[["vix", "vix3m", "rv_fwd", "pnl"]].assign(
        contango_filter=np.where(contango.loc[sub.nsmallest(5, "pnl").index], "traded", "skipped"))
    worst.index = worst.index.strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sub["pnl"].cumsum(), label="always short")
    ax.plot(sub["pnl"].where(contango, 0).cumsum(), label="short only in contango")
    ax.set(title=f"Cumulative P&L of 1-month short variance (vega notional 1, cost {COST} vol pt)",
           ylabel="vol points")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "vrp_pnl.png", dpi=120)
    plt.close(fig)

    return f"""## 3. Alpha validation: the variance risk premium

Sell a 1-month variance swap at the VIX level every 21 trading days (non-overlapping), P&L =
(VIX^2 - RV^2) / (2 VIX) per unit vega notional, minus {COST} vol pt cost. Rules are fixed in
advance (no parameter search). Sample: {sub.index.min():%b %Y} - {sub.index.max():%b %Y} (VIX3M starts 2006).

Since 1990 the always-short trade earned {long_run['mean_pnl']:.2f} vol pts per month (t = {long_run['t_stat']:.1f}),
but with skew {long_run['skew']:.1f}: a steady premium paid for crash risk.

{md(table)}

**Robustness to the roll day** (same rules, all 21 possible start offsets):

{md(robust)}

**Sub-periods** - always short:

{md(periods_all)}

Short only in contango:

{md(periods_con)}

**Worst months** for the always-short trade:

{md(worst.reset_index().rename(columns={"index": "date"}))}

![vrp](vrp_pnl.png)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download history and snapshot the chain")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.refresh:
        data.snapshot_chain()
    h = data.load_history(refresh=args.refresh)
    notes = data.check_history(h)
    h["vix"] = h["vix"].ffill(limit=1)

    s1 = surface_section(data.latest_chain())
    s2, target, har = forecast_section(h)
    s3 = vrp_section(h, target, har)
    note_md = "\n".join(f"- {n}" for n in notes)
    report = f"# Volatility research results\n\nData-quality notes:\n{note_md}\n\n{s1}\n{s2}\n{s3}"
    (OUT / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
