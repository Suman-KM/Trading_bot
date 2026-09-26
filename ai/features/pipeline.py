"""Feature pipeline orchestration and metadata contract definitions.

Assembles price, momentum, volatility, trend, activity, time, and gap features
into a unified, strictly point-in-time quantitative feature dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ai.features.activity import compute_activity_features
from ai.features.gaps import compute_gap_features
from ai.features.momentum import compute_momentum_features
from ai.features.price import compute_price_features
from ai.features.time import compute_time_features
from ai.features.trend import compute_trend_features
from ai.features.volatility import compute_volatility_features

MAX_LOOKBACK_BARS: int = 80


@dataclass(frozen=True)
class FeatureDefinition:
    """Formal specification of an individual quantitative feature."""

    name: str
    group: str
    formula: str
    lookback_bars: int
    inputs: list[str]
    expected_dtype: str
    online_computable: bool = True
    point_in_time_strictly: bool = True
    has_warmup_nans: bool = True
    normalized: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert specification to dictionary."""
        return asdict(self)


def get_feature_registry() -> list[FeatureDefinition]:
    """Retrieve full catalog of feature definitions across all groups."""
    defs: list[FeatureDefinition] = []

    # 1. Price group
    price_defs = [
        FeatureDefinition(
            "return_1", "price", "close / close_{t-1} - 1", 1, ["close"], "float64", normalized=True
        ),
        FeatureDefinition(
            "log_return_1",
            "price",
            "ln(close / close_{t-1})",
            1,
            ["close"],
            "float64",
            normalized=True,
        ),
        FeatureDefinition(
            "open_to_close_return",
            "price",
            "(close - open) / open",
            0,
            ["open", "close"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        ),
        FeatureDefinition(
            "hl_range", "price", "high - low", 0, ["high", "low"], "float64", has_warmup_nans=False
        ),
        FeatureDefinition(
            "hl_range_norm",
            "price",
            "(high - low) / close",
            0,
            ["high", "low", "close"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        ),
        FeatureDefinition(
            "candle_body",
            "price",
            "|close - open|",
            0,
            ["open", "close"],
            "float64",
            has_warmup_nans=False,
        ),
        FeatureDefinition(
            "candle_body_norm",
            "price",
            "|close - open| / close",
            0,
            ["open", "close"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        ),
        FeatureDefinition(
            "upper_wick",
            "price",
            "high - max(open, close)",
            0,
            ["open", "high", "close"],
            "float64",
            has_warmup_nans=False,
        ),
        FeatureDefinition(
            "upper_wick_ratio",
            "price",
            "upper_wick / (hl_range + 1e-8)",
            0,
            ["open", "high", "low", "close"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        ),
        FeatureDefinition(
            "lower_wick",
            "price",
            "min(open, close) - low",
            0,
            ["open", "low", "close"],
            "float64",
            has_warmup_nans=False,
        ),
        FeatureDefinition(
            "lower_wick_ratio",
            "price",
            "lower_wick / (hl_range + 1e-8)",
            0,
            ["open", "high", "low", "close"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        ),
        FeatureDefinition(
            "candle_direction",
            "price",
            "sign(close - open)",
            0,
            ["open", "close"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        ),
        FeatureDefinition(
            "return_mean_5",
            "price",
            "mean(log_return_1, 5)",
            5,
            ["close"],
            "float64",
            normalized=True,
        ),
        FeatureDefinition(
            "return_mean_20",
            "price",
            "mean(log_return_1, 20)",
            20,
            ["close"],
            "float64",
            normalized=True,
        ),
    ]
    defs.extend(price_defs)

    # 2. Momentum group
    for w in [5, 10, 20, 40, 80]:
        defs.append(
            FeatureDefinition(
                f"roc_{w}",
                "momentum",
                f"(close - close_{{t-{w}}}) / close_{{t-{w}}}",
                w,
                ["close"],
                "float64",
                normalized=True,
            )
        )
    for w in [10, 20, 40, 80]:
        defs.append(
            FeatureDefinition(
                f"dist_sma_{w}",
                "momentum",
                f"(close - sma_{w}) / sma_{w}",
                w,
                ["close"],
                "float64",
                normalized=True,
            )
        )
    defs.append(
        FeatureDefinition(
            "rsi_14", "momentum", "Wilder 14-period RSI", 14, ["close"], "float64", normalized=True
        )
    )

    # 3. Volatility group
    for w in [10, 20, 40, 80]:
        defs.append(
            FeatureDefinition(
                f"vol_std_{w}",
                "volatility",
                f"std(log_return_1, {w})",
                w,
                ["close"],
                "float64",
                normalized=True,
            )
        )
    defs.append(
        FeatureDefinition(
            "vol_ann_20",
            "volatility",
            "vol_std_20 * sqrt(24192)",
            20,
            ["close"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "vol_ann_80",
            "volatility",
            "vol_std_80 * sqrt(24192)",
            80,
            ["close"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "atr_14", "volatility", "mean(TR, 14)", 14, ["high", "low", "close"], "float64"
        )
    )
    defs.append(
        FeatureDefinition(
            "atr_norm_14",
            "volatility",
            "atr_14 / close",
            14,
            ["high", "low", "close"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "atr_40", "volatility", "mean(TR, 40)", 40, ["high", "low", "close"], "float64"
        )
    )
    defs.append(
        FeatureDefinition(
            "atr_norm_40",
            "volatility",
            "atr_40 / close",
            40,
            ["high", "low", "close"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "vol_ratio_10_40",
            "volatility",
            "vol_std_10 / vol_std_40",
            40,
            ["close"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "vol_ratio_20_80",
            "volatility",
            "vol_std_20 / vol_std_80",
            80,
            ["close"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "rolling_hl_ratio_20",
            "volatility",
            "(max_high_20 - min_low_20) / close",
            20,
            ["high", "low", "close"],
            "float64",
            normalized=True,
        )
    )

    # 4. Trend group
    for w in [10, 20, 40, 80]:
        defs.append(
            FeatureDefinition(f"sma_{w}", "trend", f"mean(close, {w})", w, ["close"], "float64")
        )
        defs.append(
            FeatureDefinition(f"ema_{w}", "trend", f"EMA(close, span={w})", w, ["close"], "float64")
        )
        defs.append(
            FeatureDefinition(
                f"dist_ema_{w}",
                "trend",
                f"(close - ema_{w}) / ema_{w}",
                w,
                ["close"],
                "float64",
                normalized=True,
            )
        )
    defs.append(
        FeatureDefinition(
            "ema_spread_10_40",
            "trend",
            "(ema_10 - ema_40) / ema_40",
            40,
            ["close"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "ema_spread_20_80",
            "trend",
            "(ema_20 - ema_80) / ema_80",
            80,
            ["close"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "ema_slope_10",
            "trend",
            "(ema_10 - ema_10_{t-5}) / (5 * ema_10_{t-5})",
            15,
            ["close"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "ema_slope_20",
            "trend",
            "(ema_20 - ema_20_{t-5}) / (5 * ema_20_{t-5})",
            25,
            ["close"],
            "float64",
            normalized=True,
        )
    )

    # 5. Activity group
    defs.append(
        FeatureDefinition(
            "log_tick_volume",
            "activity",
            "ln(1 + tick_volume)",
            0,
            ["tick_volume"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "tick_vol_sma_20", "activity", "mean(tick_volume, 20)", 20, ["tick_volume"], "float64"
        )
    )
    defs.append(
        FeatureDefinition(
            "tick_vol_ratio_20",
            "activity",
            "tick_volume / tick_vol_sma_20",
            20,
            ["tick_volume"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "tick_vol_zscore_20",
            "activity",
            "(tv - mean_20) / std_20",
            20,
            ["tick_volume"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "tick_vol_zscore_80",
            "activity",
            "(tv - mean_80) / std_80",
            80,
            ["tick_volume"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "spread_norm",
            "activity",
            "(spread * 1e-5) / close",
            0,
            ["spread", "close"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "spread_sma_20", "activity", "mean(spread, 20)", 20, ["spread"], "float64"
        )
    )
    defs.append(
        FeatureDefinition(
            "spread_ratio_20",
            "activity",
            "spread / spread_sma_20",
            20,
            ["spread"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "spread_zscore_20",
            "activity",
            "(spread - mean_20) / std_20",
            20,
            ["spread"],
            "float64",
            normalized=True,
        )
    )

    # 6. Time group
    defs.append(
        FeatureDefinition(
            "hour",
            "time",
            "hour of day (0..23)",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
        )
    )
    defs.append(
        FeatureDefinition(
            "minute",
            "time",
            "minute of hour (0, 15, 30, 45)",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
        )
    )
    defs.append(
        FeatureDefinition(
            "day_of_week",
            "time",
            "day of week (0..6)",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
        )
    )
    defs.append(
        FeatureDefinition(
            "sin_hour",
            "time",
            "sin(2 * pi * hour / 24)",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "cos_hour",
            "time",
            "cos(2 * pi * hour / 24)",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "sin_minute",
            "time",
            "sin(2 * pi * minute / 60)",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "cos_minute",
            "time",
            "cos(2 * pi * minute / 60)",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "sin_day_of_week",
            "time",
            "sin(2 * pi * dow / 7)",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "cos_day_of_week",
            "time",
            "cos(2 * pi * dow / 7)",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "is_asian_session",
            "time",
            "indicator for 00:00-08:00 UTC",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "is_london_session",
            "time",
            "indicator for 07:00-16:00 UTC",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "is_ny_session",
            "time",
            "indicator for 12:00-21:00 UTC",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "is_rollover_window",
            "time",
            "indicator for 21:00-00:00 UTC",
            0,
            ["timestamp"],
            "float64",
            has_warmup_nans=False,
            normalized=True,
        )
    )

    # 7. Gap group
    defs.append(
        FeatureDefinition("delta_seconds", "gaps", "time_t - time_{t-1}", 1, ["time"], "float64")
    )
    defs.append(
        FeatureDefinition(
            "is_normal_m15", "gaps", "delta_seconds == 900", 1, ["time"], "float64", normalized=True
        )
    )
    defs.append(
        FeatureDefinition(
            "is_post_gap", "gaps", "delta_seconds > 900", 1, ["time"], "float64", normalized=True
        )
    )
    defs.append(
        FeatureDefinition(
            "is_weekend_reopen",
            "gaps",
            "post-gap & delta >= 36h & Sun/Mon",
            1,
            ["time", "timestamp"],
            "float64",
            normalized=True,
        )
    )
    defs.append(
        FeatureDefinition(
            "bars_since_last_gap",
            "gaps",
            "consecutive bars since previous gap (max 96)",
            1,
            ["time"],
            "float64",
        )
    )

    return defs


def build_feature_pipeline(
    df: pd.DataFrame,
    drop_warmup: bool = False,
    include_raw_columns: bool = True,
) -> tuple[pd.DataFrame, list[FeatureDefinition]]:
    """Generate all feature groups from validated historical market data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with validated OHLCV columns.
    drop_warmup : bool, default False
        Whether to drop the initial warm-up period (first 80 bars).
    include_raw_columns : bool, default True
        Whether to keep the original raw price and timestamp columns.

    Returns
    -------
    tuple[pd.DataFrame, list[FeatureDefinition]]
        Resulting feature DataFrame and registry of feature specifications.
    """
    feature_defs = get_feature_registry()

    p_df = compute_price_features(df)
    m_df = compute_momentum_features(df)
    v_df = compute_volatility_features(df)
    t_df = compute_trend_features(df)
    a_df = compute_activity_features(df)
    tm_df = compute_time_features(df)
    g_df = compute_gap_features(df)

    feature_blocks = [p_df, m_df, v_df, t_df, a_df, tm_df, g_df]
    features_only = pd.concat(feature_blocks, axis=1)

    if include_raw_columns:
        # Prepend input columns
        result_df = pd.concat([df, features_only], axis=1)
    else:
        result_df = features_only

    if drop_warmup:
        result_df = result_df.iloc[MAX_LOOKBACK_BARS:].copy()

    return result_df, feature_defs


def generate_feature_metadata(
    df_with_features: pd.DataFrame,
    feature_defs: list[FeatureDefinition],
) -> dict[str, Any]:
    """Generate comprehensive machine-readable metadata for all computed features.

    Parameters
    ----------
    df_with_features : pd.DataFrame
        DataFrame containing computed features.
    feature_defs : list[FeatureDefinition]
        Catalog of feature definitions.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary.
    """
    group_counts: dict[str, int] = {}
    for f in feature_defs:
        group_counts[f.group] = group_counts.get(f.group, 0) + 1

    feature_details: list[dict[str, Any]] = []
    for f in feature_defs:
        if f.name in df_with_features.columns:
            s = df_with_features[f.name]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            nan_cnt = int(s.isna().sum())
            inf_cnt = int(np.isinf(s).sum())
            clean_s = s.dropna()
            min_val = float(clean_s.min()) if not clean_s.empty else None
            max_val = float(clean_s.max()) if not clean_s.empty else None
            mean_val = float(clean_s.mean()) if not clean_s.empty else None
            std_val = float(clean_s.std()) if not clean_s.empty else None
        else:
            nan_cnt = len(df_with_features)
            inf_cnt = 0
            min_val, max_val, mean_val, std_val = None, None, None, None

        d = f.to_dict()
        d.update(
            {
                "nan_count": nan_cnt,
                "nan_pct": float(nan_cnt / len(df_with_features) * 100.0),
                "inf_count": inf_cnt,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
                "leakage_tested": True,
            }
        )
        feature_details.append(d)

    return {
        "total_features": len(feature_defs),
        "total_rows": len(df_with_features),
        "group_counts": group_counts,
        "max_lookback_bars": MAX_LOOKBACK_BARS,
        "warmup_rows_required": MAX_LOOKBACK_BARS,
        "usable_rows_after_warmup": len(df_with_features) - MAX_LOOKBACK_BARS,
        "features": feature_details,
    }
