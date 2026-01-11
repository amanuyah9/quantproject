#!/usr/bin/env python3
"""
Build dataset for event-impact modeling.

Reads:
 - data/events.csv with columns: event_time_et (ISO string), event_name, actual (optional), forecast (optional)
 - data/spy_5m.csv with columns: timestamp_et (ISO string), open, high, low, close, volume

Writes:
 - outputs/dataset.parquet

Requirements implemented (summary):
 - All timestamps parsed/normalized to America/New_York timezone.
 - Each event is aligned to the next available 5-min bar at or after event_time_et.
 - Forward returns computed (close-to-close) for 30m, 2h, 1d horizons.
 - Realized volatility proxy: rolling 20 trading-day std of daily returns (from daily closes
   or approximated from 5-min bars), then scaled to horizons with sqrt(time).
 - Impact labels for 30m and 2h using threshold 1.5 * sigma_horizon.
 - Regime buckets based on sigma_1d quantiles (low/med/high).
 - Features: pre_event_return_1h, pre_event_return_1d (previous day return), surprise = actual - forecast
 - Robust missing handling: events without required forward bars or sigma are dropped and logged.
 - Final dataset saved to outputs/dataset.parquet and a short summary printed.
"""
from __future__ import annotations

import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Use zoneinfo for modern TZ handling
try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore

TZ_ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def parse_timestamp_series_to_et(series: pd.Series) -> pd.Series:
    """
    Parse a series of timestamp strings to timezone-aware datetime in America/New_York.
    If input timestamps are naive, assume they are ET and localize accordingly.
    If they are timezone-aware, convert to ET.
    """
    dt = pd.to_datetime(series, utc=False)
    # pandas keeps tzinfo in .dt.tz
    if dt.dt.tz is None:
        # naive -> localize to ET
        dt = dt.dt.tz_localize(TZ_ET)
    else:
        # aware -> convert to ET
        dt = dt.dt.tz_convert(TZ_ET)
    return dt


def load_data(events_csv: Path, spy_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(events_csv, dtype={"event_name": str})
    spy = pd.read_csv(spy_csv)
    # Parse timestamps
    if "event_time_et" not in events.columns:
        raise KeyError("events.csv must contain 'event_time_et' column")
    if "timestamp_et" not in spy.columns:
        raise KeyError("spy_5m.csv must contain 'timestamp_et' column")

    events["event_time_et"] = parse_timestamp_series_to_et(events["event_time_et"])
    spy["timestamp_et"] = parse_timestamp_series_to_et(spy["timestamp_et"])

    # Sort
    events = events.sort_values("event_time_et").reset_index(drop=True)
    spy = spy.sort_values("timestamp_et").reset_index(drop=True)

    # set index for fast searching
    spy = spy.set_index("timestamp_et", drop=False)
    return events, spy


def get_next_bar_index(spy_index: pd.DatetimeIndex, t: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    Return the timestamp of the first bar at or after time t.
    """
    pos = spy_index.searchsorted(t, side="left")
    if pos >= len(spy_index):
        return None
    return spy_index[pos]


def get_bar_at_or_after(spy_index: pd.DatetimeIndex, t: pd.Timestamp, horizon: timedelta) -> Optional[pd.Timestamp]:
    """
    Return the timestamp of the first bar at or after t + horizon.
    """
    target = t + horizon
    return get_next_bar_index(spy_index, target)


def daily_close_from_5m(spy: pd.DataFrame) -> pd.Series:
    """
    Approximate daily close series by taking the last 5-min close of each calendar date
    in ET timezone.
    Returns a pd.Series indexed by date (Timestamp at midnight ET) with 'close'.
    """
    # Make sure timestamp index is tz-aware ET
    idx = spy.index
    # Group by date in ET timezone
    dates = idx.tz_convert(TZ_ET).date
    # Use groupby on dates to get last close
    spy_with_date = spy.copy()
    spy_with_date["_date"] = pd.to_datetime(dates)
    daily = spy_with_date.groupby("_date", as_index=True)["close"].last()
    daily.index = pd.to_datetime(daily.index).tz_localize(TZ_ET)
    daily = daily.sort_index()
    return daily


def compute_rolling_sigma_daily(daily_close: pd.Series, window: int = 20, min_periods: int = 5) -> pd.Series:
    """
    Compute daily returns and rolling std dev (sigma) of daily returns.
    Returns sigma aligned with the day of the return (i.e., sigma at day t is std of returns ending at t).
    """
    daily_returns = daily_close.pct_change().dropna()
    sigma = daily_returns.rolling(window=window, min_periods=min_periods).std()
    # sigma is in daily-return units (not annualized)
    return sigma


def attach_sigma_to_events(
    events: pd.DataFrame,
    sigma_daily: pd.Series,
) -> pd.Series:
    """
    For each event, find the most recent sigma_daily value available at or before the event date (by day).
    Returns a pd.Series indexed like events with sigma_1d (daily sigma).
    """
    # sigma_daily indexed by day's timestamp at midnight ET
    # Map each event to sigma on same day (use the last available sigma at or before event date)
    # Convert event times to date aligned index for sigma lookup
    event_days = events["event_time_et"].dt.normalize()  # midnight ET, tz-aware
    # Ensure sigma index is normalized
    sigma_index = sigma_daily.index.normalize()
    # Build an ndarray of sigma lookup using searchsorted
    sigma_sorted_index = sigma_daily.index
    sigma_vals = []
    for ev_time in events["event_time_et"]:
        day_mid = ev_time.normalize()
        # We want sigma at or before that day.
        pos = sigma_sorted_index.searchsorted(day_mid, side="right") - 1
        if pos >= 0:
            sigma_vals.append(sigma_daily.iloc[pos])
        else:
            sigma_vals.append(np.nan)
    return pd.Series(sigma_vals, index=events.index)


def horizon_in_days(delta: timedelta) -> float:
    return delta.total_seconds() / (24 * 3600)


def safe_div(a: float, b: float) -> Optional[float]:
    try:
        return a / b
    except Exception:
        return None


def main():
    base = Path(".")
    events_csv = base / "data" / "events.csv"
    spy_csv = base / "data" / "spy_5m.csv"
    out_dir = base / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dataset.parquet"

    events, spy = load_data(events_csv, spy_csv)
    logging.info(f"Loaded {len(events)} events and {len(spy)} 5-min bars")

    spy_index = spy.index  # DatetimeIndex of bar timestamps tz-aware ET

    # Prepare daily close series and sigma
    daily_close = daily_close_from_5m(spy)
    if daily_close.empty:
        logging.warning("No daily closes could be extracted from 5-min data; sigma will be NaN")
    sigma_daily = compute_rolling_sigma_daily(daily_close, window=20, min_periods=5)

    # For each event, align to next available 5-min bar at or after event_time_et
    aligned_bars = []
    aligned_timestamps = []
    missing_alignment = 0
    for ev_time in events["event_time_et"]:
        tio = get_next_bar_index(spy_index, ev_time)
        if tio is None:
            aligned_bars.append(None)
            aligned_timestamps.append(pd.NaT)
            missing_alignment += 1
        else:
            aligned_bars.append(spy_index.get_loc(tio))
            aligned_timestamps.append(tio)
    events["aligned_bar_ts"] = pd.to_datetime(aligned_timestamps)
    events["aligned_bar_ts"] = events["aligned_bar_ts"].dt.tz_localize(TZ_ET, ambiguous="NaT") if events["aligned_bar_ts"].dt.tz is None else events["aligned_bar_ts"]

    logging.info(f"Events with no available alignment bar at/after event_time_et: {missing_alignment}")

    # Prepare horizons
    horizons = {
        "30m": timedelta(minutes=30),
        "2h": timedelta(hours=2),
        "1d": timedelta(days=1),
    }

    # For each event, find the bar at or after event_time + horizon
    forward_ts = {h: [] for h in horizons.keys()}
    for i, ev_time in enumerate(events["event_time_et"]):
        for h_name, h_delta in horizons.items():
            if pd.isna(events.loc[i, "aligned_bar_ts"]):
                forward_ts[h_name].append(pd.NaT)
                continue
            t_align = events.loc[i, "aligned_bar_ts"]
            t_forward = get_bar_at_or_after(spy_index, t_align, h_delta)
            forward_ts[h_name].append(t_forward if t_forward is not None else pd.NaT)

    for h_name in horizons.keys():
        events[f"forward_{h_name}_ts"] = pd.to_datetime(forward_ts[h_name])
        # ensure tz-aware
        events[f"forward_{h_name}_ts"] = events[f"forward_{h_name}_ts"].dt.tz_localize(TZ_ET, ambiguous="NaT") if events[f"forward_{h_name}_ts"].dt.tz is None else events[f"forward_{h_name}_ts"]

    # Compute forward returns close-to-close: (close_forward / close_align) - 1
    def get_close_at(ts: pd.Timestamp) -> Optional[float]:
        if pd.isna(ts):
            return np.nan
        try:
            return float(spy.loc[ts, "close"])
        except KeyError:
            # If exact timestamp not found in index (shouldn't happen), try to locate nearest
            pos = spy_index.searchsorted(ts, side="left")
            if pos >= len(spy_index):
                return np.nan
            return float(spy.iloc[pos]["close"])

    for h_name in horizons.keys():
        vals = []
        for i in events.index:
            t_align = events.loc[i, "aligned_bar_ts"]
            t_forward = events.loc[i, f"forward_{h_name}_ts"]
            if pd.isna(t_align) or pd.isna(t_forward):
                vals.append(np.nan)
                continue
            close_align = get_close_at(t_align)
            close_forward = get_close_at(t_forward)
            if pd.isna(close_align) or pd.isna(close_forward) or close_align == 0:
                vals.append(np.nan)
            else:
                vals.append((close_forward / close_align) - 1.0)
        events[f"ret_{h_name}"] = vals

    # Compute sigma_1d attached to events
    events["sigma_1d"] = attach_sigma_to_events(events, sigma_daily)
    # If sigma_1d is NaN but we can try a fallback: use rolling std computed from intraday daily returns if any
    num_missing_sigma = events["sigma_1d"].isna().sum()
    if num_missing_sigma:
        logging.info(f"{num_missing_sigma} events missing sigma_1d from rolling daily returns")

    # For each horizon compute sigma_h = sigma_1d * sqrt(horizon_in_days)
    for h_name, h_delta in horizons.items():
        events[f"sigma_{h_name}"] = events["sigma_1d"] * math.sqrt(horizon_in_days(h_delta))

    # Create impact labels for 30m and 2h
    for label_h in ["30m", "2h"]:
        ret_col = f"ret_{label_h}"
        sigma_col = f"sigma_{label_h}"
        impact_col = f"impact_{label_h}"
        # threshold = 1.5 * sigma_h
        events[impact_col] = np.where(
            events[ret_col].abs() > 1.5 * events[sigma_col], 1, 0
        )
        # If either ret or sigma is NaN, set label to NaN (we'll drop later)
        events.loc[events[ret_col].isna() | events[sigma_col].isna(), impact_col] = np.nan

    # Regime buckets based on sigma_1d quantiles (use available sigma_1d values)
    sigma_vals = events["sigma_1d"].dropna()
    if not sigma_vals.empty:
        q_lower = sigma_vals.quantile(0.33)
        q_upper = sigma_vals.quantile(0.66)

        def regime_label(s):
            if pd.isna(s):
                return pd.NA
            if s <= q_lower:
                return "low"
            if s <= q_upper:
                return "med"
            return "high"

        events["regime_1d"] = events["sigma_1d"].apply(regime_label)
    else:
        events["regime_1d"] = pd.NA

    # Feature engineering
    # pre_event_return_1h: return from -1h to event aligned bar: (close_align / close_t_minus_1h) - 1
    pre1h_vals = []
    for i in events.index:
        t_align = events.loc[i, "aligned_bar_ts"]
        if pd.isna(t_align):
            pre1h_vals.append(np.nan)
            continue
        t_minus_1h = t_align - timedelta(hours=1)
        # We want the bar at or before t_minus_1h (the last available bar at/before that time)
        pos = spy_index.searchsorted(t_minus_1h, side="right") - 1
        if pos < 0:
            pre1h_vals.append(np.nan)
            continue
        t_before = spy_index[pos]
        close_before = float(spy.iloc[pos]["close"])
        close_align = get_close_at(t_align)
        if pd.isna(close_before) or pd.isna(close_align) or close_before == 0:
            pre1h_vals.append(np.nan)
        else:
            pre1h_vals.append((close_align / close_before) - 1.0)
    events["pre_event_return_1h"] = pre1h_vals

    # pre_event_return_1d: previous trading day return (daily return of previous day)
    # We'll compute daily closes and use (close_prev / close_prevprev) - 1 where close_prev is previous day close
    # Map events to previous-day daily return (i.e., the return that occurred on the previous trading day)
    daily_returns = daily_close.pct_change()
    # Build series of previous-day return for each event: find previous trading day's date, and take that day's return
    prev_day_returns = []
    daily_index = daily_returns.index.normalize()
    for ev_time in events["event_time_et"]:
        day_mid = ev_time.normalize()
        # We want the return that corresponds to the previous trading day; that is the daily return whose index == day_mid - 1 trading day
        # Find position of first daily index >= day_mid, then move back by 1 to get previous day's return
        pos = daily_index.searchsorted(day_mid, side="left") - 1
        if pos <= 0:
            prev_day_returns.append(np.nan)
        else:
            prev_day_returns.append(float(daily_returns.iloc[pos]))
    events["pre_event_return_1d"] = prev_day_returns

    # surprise if actual & forecast exist
    if "actual" in events.columns and "forecast" in events.columns:
        events["surprise"] = events["actual"] - events["forecast"]
    else:
        events["surprise"] = pd.NA

    # Handle missing data robustly: drop events without enough forward bars or missing impact labels
    initial_count = len(events)
    # Criteria for dropping:
    # - aligned bar missing
    # - forward bar missing for any horizon of interest (we need at least 30m and 2h for labels; 1d used as feature)
    drop_mask = (
        events["aligned_bar_ts"].isna()
        | events["forward_30m_ts"].isna()
        | events["forward_2h_ts"].isna()
        # also require sigma for 30m and 2h not NaN (to compute labels)
        | events["sigma_30m"].isna()
        | events["sigma_2h"].isna()
        # require the forward returns present
        | events["ret_30m"].isna()
        | events["ret_2h"].isna()
    )
    dropped_events = events[drop_mask].copy()
    kept_events = events[~drop_mask].copy()
    num_dropped = drop_mask.sum()

    # Log counts for other missing features (not fatal): pre_event_return_1h, pre_event_return_1d, surprise can be NaN
    num_missing_pre1h = events["pre_event_return_1h"].isna().sum()
    num_missing_pre1d = events["pre_event_return_1d"].isna().sum()
    num_missing_surprise = events["surprise"].isna().sum() if "surprise" in events.columns else 0

    logging.info(f"Dropping {num_dropped} events due to missing alignment/forward bars or sigma")
    logging.info(f"Missing pre_event_return_1h: {num_missing_pre1h}, pre_event_return_1d: {num_missing_pre1d}, surprise: {num_missing_surprise}")

    # Final dataset
    df_final = kept_events.copy()

    # Reformat columns: ensure timestamps are timezone-aware and formatted consistently
    ts_cols = [c for c in df_final.columns if c.endswith("_ts") or c == "event_time_et" or c == "aligned_bar_ts"]
    for c in ts_cols:
        df_final[c] = pd.to_datetime(df_final[c]).dt.tz_convert(TZ_ET)

    # Save to parquet
    df_final.to_parquet(out_file, index=False)
    logging.info(f"Saved final dataset with {len(df_final)} rows to {out_file}")

    # Print short summary
    print("SUMMARY")
    print("-------")
    print(f"Number of events processed (input): {initial_count}")
    print(f"Number dropped for missing price windows / sigma: {num_dropped}")
    # Label prevalence for each horizon
    for h in ["30m", "2h"]:
        col = f"impact_{h}"
        if col in df_final.columns:
            # Note: impact labels are 0/1
            n_pos = int((df_final[col] == 1).sum())
            n_total = int(df_final[col].notna().sum())
            prevalence = n_pos / n_total if n_total > 0 else float("nan")
            print(f"Label prevalence {col}: {n_pos}/{n_total} = {prevalence:.3f}")
        else:
            print(f"Label {col} not present in final dataset")

    # Also print counts of regimes
    if "regime_1d" in df_final.columns:
        regime_counts = df_final["regime_1d"].value_counts(dropna=False)
        print("Regime counts:")
        for lab, cnt in regime_counts.items():
            print(f"  {lab}: {cnt}")

    # End
    logging.info("Done.")


if __name__ == "__main__":
    main()