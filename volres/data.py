"""Market data: S&P 500 OHLC, VIX indices, and SPY option-chain snapshots (via yfinance).

Everything is cached under data/ so research runs are reproducible offline.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")


def _download(ticker: str, start: str = "1990-01-01") -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].rename(columns=str.lower)


def load_history(refresh: bool = False) -> pd.DataFrame:
    """Daily frame: S&P 500 OHLC + VIX and VIX3M closes (in vol points)."""
    path = DATA / "history.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    spx = _download("^GSPC")
    out = spx.join(_download("^VIX")["close"].rename("vix"), how="left")
    out = out.join(_download("^VIX3M")["close"].rename("vix3m"), how="left")
    DATA.mkdir(exist_ok=True)
    out.to_csv(path)
    return out


def check_history(df: pd.DataFrame) -> list[str]:
    """Data-quality notes that change how the data may be used."""
    notes = []
    stale_open = (df["open"] == df["close"].shift()).groupby(df.index.year).mean()
    bad_years = stale_open[stale_open > 0.5].index
    if len(bad_years):
        notes.append(
            f"S&P 500 open equals the previous close on >50% of days in {bad_years.min()}-{bad_years.max()}: "
            "opens are not real before then, so intraday variance uses the high-low (Parkinson) "
            "estimator, which never touches the open"
        )
    if (df["high"] < df["low"]).any():
        notes.append("rows with high < low")
    gaps = df["vix"].isna().sum()
    if gaps:
        notes.append(f"{gaps} days with no VIX close (forward-filled at most 1 day)")
    return notes


def snapshot_chain(ticker: str = "SPY", max_days: int = 400) -> Path:
    """Save today's option chain (all expiries up to `max_days`) with the spot price."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    today = pd.Timestamp(date.today())
    frames = []
    for exp in tk.options:
        if (pd.Timestamp(exp) - today).days > max_days:
            break
        chain = tk.option_chain(exp)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            frames.append(df.assign(type=kind, expiry=exp))
    cols = ["expiry", "type", "strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"]
    # 13-week T-bill yield (percent) as the discount rate for put-call parity
    rate = float(yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]) / 100
    out = pd.concat(frames)[cols].assign(spot=spot, rate=rate, asof=today.date().isoformat())
    path = DATA / f"chain_{ticker}_{today:%Y%m%d}.csv"
    DATA.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    return path


def latest_chain(ticker: str = "SPY") -> pd.DataFrame:
    files = sorted(DATA.glob(f"chain_{ticker}_*.csv"))
    if not files:
        raise FileNotFoundError("no option-chain snapshot; run with --snapshot first")
    return pd.read_csv(files[-1])


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()
