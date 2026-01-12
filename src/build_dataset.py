#!/usr/bin/env python3
"""
Build dataset for event-impact modeling (v1: intraday only, correct horizons).

Reads:
 - data/events.csv: event_time_et (ISO string), event_name, actual (opt), forecast (opt)
 - data/spy_5m.csv: timestamp_et (ISO string), open, high, low, close, volume

Writes:
 - outputs/dataset.parquet

Key design choices (to avoid silent finance bugs):
 - Use 5-min BAR COUNTS for horizons: 30m = +6 bars, 2h = +24 bars (no calendar timedelta).
 - Compute daily closes and daily volatility using REGULAR TRADING HOURS (RTH) only (09:30-16:00 ET).
 - Scale daily sigma to intraday using TRADING MINUTES (390 min/session), not 24-hour days.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore

TZ_ET = ZoneInfo("America/New_York")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_to_et(series: pd.Series) -> pd.Series:
    """
    Robust timestamp parsing (DST/mixed offsets safe).

    - If strings include timezone offsets (e.g., -05:00 and -04:00) or Z:
        parse with utc=True so pandas produces a single tz-aware dtype,
        then convert to America/New_York.

    - If strings are naive (no tz info):
        parse as naive then localize to America/New_York.
    """
    s = series.astype(str)

    # detect timezone info in strings
    has_tz_info = s.str.contains(r"(?:Z|[+-]\d{2}:\d{2})", regex=True, na=False).any()

    if has_tz_info:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        return dt.dt.tz_convert(TZ_ET)
    else:
        dt = pd.to_datetime(s, errors="coerce")
        return dt.dt.tz_localize(TZ_ET)



def filter_rth(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Keep only regular trading hours for US equities: 09:30–16:00 ET."""
    ts = df[ts_col]
    # between_time works on index; temporarily set index
    tmp = df.set_index(ts_col, drop=False)
    tmp = tmp.sort_index()
    tmp = tmp.between_time("09:30", "16:00", inclusive="both")
    return tmp.reset_index(drop=True)


def daily_close_rth(spy_rth: pd.DataFrame) -> pd.Series:
    """Daily close = last RTH close per trading day."""
    spy_rth = spy_rth.copy()
    spy_rth["date"] = spy_rth["timestamp_et"].dt.normalize()
    daily = spy_rth.groupby("date")["close"].last().sort_index()
    return daily


def rolling_sigma_1d(daily_close: pd.Series, window: int = 20, min_periods: int = 10) -> pd.Series:
    """Rolling std of daily returns (not annualized)."""
    daily_ret = daily_close.pct_change()
    sigma = daily_ret.rolling(window=window, min_periods=min_periods).std()
    return sigma


def align_event_to_next_bar(spy_index: pd.DatetimeIndex, t: pd.Timestamp) -> Optional[int]:
    """Return integer position of the first bar at or after event time."""
    pos = spy_index.searchsorted(t, side="left")
    if pos >= len(spy_index):
        return None
    return int(pos)


def main() -> None:
    base = Path(".")
    events_csv = base / "data" / "events.csv"
    spy_csv = base / "data" / "spy_5m.csv"
    out_dir = base / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dataset.parquet"

    events = pd.read_csv(events_csv)
    spy = pd.read_csv(spy_csv)

    if "event_time_et" not in events.columns:
        raise KeyError("events.csv must contain event_time_et")
    if "timestamp_et" not in spy.columns:
        raise KeyError("spy_5m.csv must contain timestamp_et")

    events["event_time_et"] = parse_to_et(events["event_time_et"])
    spy["timestamp_et"] = parse_to_et(spy["timestamp_et"])

    events = events.dropna(subset=["event_time_et"]).sort_values("event_time_et").reset_index(drop=True)
    spy = spy.dropna(subset=["timestamp_et"]).sort_values("timestamp_et").reset_index(drop=True)

    # Keep full spy for intraday returns, but build RTH daily series for sigma/regime features
    spy_rth = filter_rth(spy, "timestamp_et")

    # Set index for fast bar lookups (use full spy bars, not just RTH, because events can be premarket)
    spy = spy.set_index("timestamp_et", drop=False).sort_index()
    spy_index = spy.index

    # Daily close + rolling sigma_1d from RTH
    daily_close = daily_close_rth(spy_rth)
    sigma_1d_series = rolling_sigma_1d(daily_close)

    # Map sigma_1d to each event using last available sigma strictly BEFORE the event day
    # (prevents using same-day close info that occurs after 08:30 releases)
    sigma_dates = sigma_1d_series.index
    sigma_vals = []
    for t in events["event_time_et"]:
        day = t.normalize()
        # find last sigma date < event day
        pos = sigma_dates.searchsorted(day, side="left") - 1
        sigma_vals.append(float(sigma_1d_series.iloc[pos]) if pos >= 0 else np.nan)
    events["sigma_1d"] = sigma_vals

    # Intraday sigma scaling using trading minutes (390 min/day)
    events["sigma_30m"] = events["sigma_1d"] * np.sqrt(30.0 / 390.0)
    events["sigma_2h"] = events["sigma_1d"] * np.sqrt(120.0 / 390.0)

    # Regime buckets by sigma_1d quantiles
    valid_sigma = events["sigma_1d"].dropna()
    if len(valid_sigma) >= 30:
        q1, q2 = valid_sigma.quantile([0.33, 0.66]).values
        def reg(s: float) -> str:
            if np.isnan(s):
                return "NA"
            if s <= q1:
                return "low"
            if s <= q2:
                return "med"
            return "high"
        events["regime_1d"] = events["sigma_1d"].apply(reg)
    else:
        events["regime_1d"] = "NA"

    # Horizon definitions by bar count
    HORIZONS = {"30m": 6, "2h": 24}

    aligned_pos = []
    aligned_ts = []
    for t in events["event_time_et"]:
        pos = align_event_to_next_bar(spy_index, t)
        aligned_pos.append(pos if pos is not None else np.nan)
        aligned_ts.append(spy_index[pos] if pos is not None else pd.NaT)

    events["aligned_pos"] = aligned_pos
    events["aligned_bar_ts"] = aligned_ts

    # Forward timestamps/returns
    for name, n_bars in HORIZONS.items():
        f_ts = []
        f_ret = []
        for pos in events["aligned_pos"]:
            if np.isnan(pos):
                f_ts.append(pd.NaT)
                f_ret.append(np.nan)
                continue
            pos = int(pos)
            fpos = pos + n_bars
            if fpos >= len(spy):
                f_ts.append(pd.NaT)
                f_ret.append(np.nan)
                continue
            t0 = spy_index[pos]
            t1 = spy_index[fpos]
            c0 = float(spy.loc[t0, "close"])
            c1 = float(spy.loc[t1, "close"])
            f_ts.append(t1)
            f_ret.append((c1 / c0) - 1.0 if c0 != 0 else np.nan)
        events[f"forward_{name}_ts"] = f_ts
        events[f"ret_{name}"] = f_ret

    # Pre-event return 1h (12 bars back)
    pre1h = []
    for pos in events["aligned_pos"]:
        if np.isnan(pos):
            pre1h.append(np.nan)
            continue
        pos = int(pos)
        bpos = pos - 12
        if bpos < 0:
            pre1h.append(np.nan)
            continue
        t_before = spy_index[bpos]
        t_align = spy_index[pos]
        c_before = float(spy.loc[t_before, "close"])
        c_align = float(spy.loc[t_align, "close"])
        pre1h.append((c_align / c_before) - 1.0 if c_before != 0 else np.nan)
    events["pre_event_return_1h"] = pre1h

    # Surprise feature if available
    if "actual" in events.columns and "forecast" in events.columns:
        events["surprise"] = events["actual"] - events["forecast"]
    else:
        events["surprise"] = np.nan

    # Impact labels (vol-adjusted)
    for name in ["30m", "2h"]:
        events[f"impact_{name}"] = np.where(
            (events[f"ret_{name}"].abs() > 1.5 * events[f"sigma_{name}"]) &
            events[f"ret_{name}"].notna() &
            events[f"sigma_{name}"].notna(),
            1, 0
        )

    # Drop rows with missing essentials
    before = len(events)
    keep = (
        events["aligned_bar_ts"].notna() &
        events["ret_30m"].notna() &
        events["ret_2h"].notna() &
        events["sigma_1d"].notna()
    )
    df = events.loc[keep].copy()
    dropped = before - len(df)

    # Sanity checks (print a few rows)
    print("SANITY CHECK (first 5 rows)")
    cols = ["event_time_et", "aligned_bar_ts", "forward_30m_ts", "forward_2h_ts", "ret_30m", "ret_2h", "sigma_30m", "sigma_2h"]
    print(df[cols].head().to_string(index=False))

    # Verify bar increments (sample)
    sample = df.head(10)
    for name, n_bars in HORIZONS.items():
        ok = 0
        for _, r in sample.iterrows():
            pos0 = spy_index.get_loc(r["aligned_bar_ts"])
            pos1 = spy_index.get_loc(r[f"forward_{name}_ts"])
            ok += int((pos1 - pos0) == n_bars)
        print(f"Sanity: {name} horizon bars correct in sample: {ok}/{len(sample)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False)

    print("\nSUMMARY")
    print("-------")
    print(f"Input events: {before}")
    print(f"Dropped (missing essentials): {dropped}")
    for name in ["30m", "2h"]:
        prev = df[f"impact_{name}"].mean()
        print(f"impact_{name} prevalence: {prev:.3f}")

    logging.info(f"Saved dataset: {out_file} rows={len(df)}")


if __name__ == "__main__":
    main()