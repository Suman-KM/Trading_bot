"""Causal point-in-time alignment utilities for exogenous data.

Enforces strict non-anticipative mapping of discrete exogenous events and
macro releases onto continuous OHLCV bar timelines with deterministic latency.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from ai.data.exogenous.schema import AlignmentConfig, AlignmentResult, ExogenousDataPoint


def compute_first_usable_bar(
    information_timestamp: datetime,
    config: AlignmentConfig | None = None,
) -> datetime:
    """Calculate the earliest bar timestamp permitted to observe an event.

    Given an external event known at `information_timestamp`, determines
    the exact bar timestamp where this information is first causal.

    Parameters
    ----------
    information_timestamp : datetime
        Exact UTC timestamp when the information was published/known.
    config : AlignmentConfig, optional
        Alignment settings (timeframe, open vs close timestamp, latency).

    Returns
    -------
    datetime
        UTC timestamp of the earliest bar allowed to incorporate this information.
    """
    if config is None:
        config = AlignmentConfig()

    if information_timestamp.tzinfo is None:
        information_timestamp = information_timestamp.replace(tzinfo=UTC)

    # Physical arrival after network/ingestion latency buffer
    effective_time = information_timestamp + timedelta(seconds=config.latency_buffer_seconds)

    tf_seconds = config.bar_timeframe_minutes * 60
    epoch_sec = effective_time.timestamp()

    if config.bar_timestamp_is_open:
        # Each bar [T_open, T_open + tf) closes at T_open + tf.
        # A bar can only use data if its close time >= effective_time.
        # Therefore: T_open + tf >= effective_time => T_open >= effective_time - tf.
        # The earliest T_open on the grid where T_open + tf >= effective_time:
        # If effective_time is exactly on a boundary, say 13:45:00,
        # then the bar opening at 13:30:00 closes at 13:45:00 (which is >= 13:45:00).
        # In general, T_close = ceil(epoch_sec / tf_seconds) * tf_seconds.
        # T_open = T_close - tf_seconds.
        close_epoch = math.ceil(epoch_sec / tf_seconds) * tf_seconds
        open_epoch = close_epoch - tf_seconds
        return datetime.fromtimestamp(open_epoch, tz=UTC)
    else:
        # Bar timestamp represents close time
        close_epoch = math.ceil(epoch_sec / tf_seconds) * tf_seconds
        return datetime.fromtimestamp(close_epoch, tz=UTC)


def align_exogenous_events_to_bars(
    df_bars: pd.DataFrame,
    events: list[ExogenousDataPoint],
    value_attribute: str = "surprise_value",
    fill_policy: str = "forward_fill",
    config: AlignmentConfig | None = None,
    time_col: str = "timestamp",
) -> tuple[pd.Series, AlignmentResult]:
    """Align a collection of exogenous points to a bar DataFrame causally.

    Parameters
    ----------
    df_bars : pd.DataFrame
        Market DataFrame containing bar timestamps.
    events : list[ExogenousDataPoint]
        List of verified exogenous event observations.
    value_attribute : str, default "surprise_value"
        Attribute on ExogenousDataPoint to extract.
    fill_policy : str, default "forward_fill"
        Policy: "forward_fill" or "exact_bar_only".
    config : AlignmentConfig, optional
        Alignment parameters.
    time_col : str, default "timestamp"
        Timestamp column name in df_bars.

    Returns
    -------
    tuple[pd.Series, AlignmentResult]
        The aligned causal feature Series and audit report.
    """
    if config is None:
        config = AlignmentConfig()

    errors: list[str] = []
    if time_col not in df_bars.columns:
        raise ValueError(f"Time column '{time_col}' not found in df_bars.")

    bar_times = pd.to_datetime(df_bars[time_col], utc=True)
    n_bars = len(bar_times)

    if n_bars == 0:
        return pd.Series(dtype=np.float64), AlignmentResult(
            total_bars=0,
            aligned_bars=0,
            missing_bars=0,
            lookahead_violations=0,
            earliest_aligned=None,
            latest_aligned=None,
            errors=["df_bars is empty."],
        )

    # Sort events chronologically by first_usable_bar_timestamp
    sorted_events = sorted(events, key=lambda e: e.first_usable_bar_timestamp)

    # Map events to their designated first usable bar
    aligned_values = np.full(n_bars, np.nan, dtype=np.float64)
    event_idx = 0
    n_events = len(sorted_events)
    lookahead_violations = 0
    current_propagated_value = np.nan
    last_event_usable_time: datetime | None = None
    bars_since_last_event = 999999

    tf_delta = timedelta(minutes=config.bar_timeframe_minutes)
    max_lookback_delta = timedelta(minutes=config.max_lookback_bars * config.bar_timeframe_minutes)

    for i in range(n_bars):
        bar_t = bar_times.iloc[i].to_pydatetime()
        if bar_t.tzinfo is None:
            bar_t = bar_t.replace(tzinfo=UTC)

        # Bar close time
        bar_close_t = bar_t + tf_delta if config.bar_timestamp_is_open else bar_t

        # Ingest all events that became usable at or before this bar
        new_event_for_this_bar = False
        while event_idx < n_events:
            ev = sorted_events[event_idx]
            ev_usable_t = ev.first_usable_bar_timestamp
            if ev_usable_t <= bar_t:
                # Causality verification: ensure publication + buffer is strictly <= bar_close_t
                info_t = ev.information_timestamp
                buffered_info = info_t + timedelta(seconds=config.latency_buffer_seconds)
                if buffered_info > bar_close_t:
                    lookahead_violations += 1
                    errors.append(
                        f"Lookahead violation at bar {bar_t}: event {ev.series_id} published at "
                        f"{info_t} (buffer={buffered_info}) > bar_close {bar_close_t}."
                    )

                val = getattr(ev, value_attribute, np.nan)
                current_propagated_value = float(val) if val is not None else np.nan
                last_event_usable_time = ev_usable_t
                new_event_for_this_bar = True
                bars_since_last_event = 0
                event_idx += 1
            else:
                break

        if fill_policy == "exact_bar_only":
            # Exact bar only applies if the event's first usable bar matches this bar
            if new_event_for_this_bar and last_event_usable_time == bar_t:
                aligned_values[i] = current_propagated_value
            else:
                aligned_values[i] = np.nan
        elif fill_policy == "forward_fill":
            if not np.isnan(current_propagated_value) and last_event_usable_time is not None:
                elapsed_time = bar_t - last_event_usable_time
                if (
                    bars_since_last_event <= config.max_lookback_bars
                    and elapsed_time <= max_lookback_delta
                ):
                    aligned_values[i] = current_propagated_value
                else:
                    # Decay / expire after max lookback
                    aligned_values[i] = np.nan
                bars_since_last_event += 1
            else:
                aligned_values[i] = np.nan

    series = pd.Series(aligned_values, index=df_bars.index, name=f"exo_{value_attribute}")
    aligned_bars = int((~series.isna()).sum())
    missing_bars = n_bars - aligned_bars

    earliest_aligned = None
    latest_aligned = None
    if aligned_bars > 0:
        valid_indices = series.dropna().index
        earliest_aligned = str(bar_times.loc[valid_indices[0]])
        latest_aligned = str(bar_times.loc[valid_indices[-1]])

    result = AlignmentResult(
        total_bars=n_bars,
        aligned_bars=aligned_bars,
        missing_bars=missing_bars,
        lookahead_violations=lookahead_violations,
        earliest_aligned=earliest_aligned,
        latest_aligned=latest_aligned,
        errors=errors,
    )

    return series, result
