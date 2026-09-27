"""Label pipeline orchestration and metadata contract definitions.

Assembles forward returns, continuous log returns, volatility-adjusted returns,
and ternary directional classification targets into a unified target dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ai.labels.direction import DEFAULT_FIXED_THRESHOLDS, compute_direction_labels
from ai.labels.returns import DEFAULT_HORIZONS, compute_future_returns


@dataclass(frozen=True)
class LabelDefinition:
    """Formal specification of an individual quantitative prediction target."""

    name: str
    horizon_bars: int
    horizon_minutes: int
    label_type: str
    formula: str
    threshold: float | str
    classes: dict[float, str] | None
    source_columns: list[str]
    uses_future_information: bool = True
    end_of_data_nans: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert specification to dictionary."""
        return asdict(self)


def get_label_registry(
    horizons: list[int] | None = None,
) -> list[LabelDefinition]:
    """Retrieve full catalog of label specifications across candidate horizons.

    Parameters
    ----------
    horizons : list[int] | None, default None
        List of horizons in bars. Defaults to [1, 4, 8, 16].

    Returns
    -------
    list[LabelDefinition]
        Catalog of label definitions.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    classes_dict = {-1.0: "SHORT", 0.0: "NEUTRAL", 1.0: "LONG"}
    defs: list[LabelDefinition] = []

    for h in horizons:
        mins = h * 15
        th = DEFAULT_FIXED_THRESHOLDS.get(h, 0.00025 * np.sqrt(h))

        # 1. Simple Forward Return
        defs.append(
            LabelDefinition(
                name=f"future_return_{h}",
                horizon_bars=h,
                horizon_minutes=mins,
                label_type="continuous_return",
                formula=f"close[t + {h}] / close[t] - 1",
                threshold="N/A",
                classes=None,
                source_columns=["close"],
                end_of_data_nans=h,
            )
        )

        # 2. Continuous Forward Log Return
        defs.append(
            LabelDefinition(
                name=f"future_log_return_{h}",
                horizon_bars=h,
                horizon_minutes=mins,
                label_type="continuous_log_return",
                formula=f"ln(close[t + {h}] / close[t])",
                threshold="N/A",
                classes=None,
                source_columns=["close"],
                end_of_data_nans=h,
            )
        )

        # 3. Volatility-Adjusted Forward Return
        defs.append(
            LabelDefinition(
                name=f"future_vol_adj_return_{h}",
                horizon_bars=h,
                horizon_minutes=mins,
                label_type="volatility_adjusted_return",
                formula=f"future_return_{h} / atr_norm_14[t]",
                threshold="N/A",
                classes=None,
                source_columns=["close", "high", "low"],
                end_of_data_nans=h,
            )
        )

        # 4. Multiclass Direction (Fixed Threshold)
        defs.append(
            LabelDefinition(
                name=f"direction_{h}",
                horizon_bars=h,
                horizon_minutes=mins,
                label_type="multiclass_direction",
                formula=f"+1 if ret > {th} else (-1 if ret < -{th} else 0)",
                threshold=th,
                classes=classes_dict,
                source_columns=["close"],
                end_of_data_nans=h,
            )
        )

        # 5. Volatility-Scaled Multiclass Direction
        defs.append(
            LabelDefinition(
                name=f"direction_vol_{h}",
                horizon_bars=h,
                horizon_minutes=mins,
                label_type="volatility_scaled_direction",
                formula=(
                    f"+1 if ret > 0.5*sqrt({h})*ATR else (-1 if ret < -0.5*sqrt({h})*ATR else 0)"
                ),
                threshold=f"0.5 * sqrt({h}) * ATR_14",
                classes=classes_dict,
                source_columns=["close", "high", "low"],
                end_of_data_nans=h,
            )
        )

    return defs


def build_label_pipeline(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    include_timestamp: bool = True,
) -> tuple[pd.DataFrame, list[LabelDefinition]]:
    """Generate all target labels across candidate prediction horizons.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with validated market data ('close', 'high', 'low', 'time', 'timestamp').
    horizons : list[int] | None, default None
        Candidate prediction horizons in bars. Defaults to [1, 4, 8, 16].
    include_timestamp : bool, default True
        Whether to include alignment timestamp columns.

    Returns
    -------
    tuple[pd.DataFrame, list[LabelDefinition]]
        Resulting target label DataFrame and registry of label specifications.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    label_defs = get_label_registry(horizons)

    returns_df = compute_future_returns(df, horizons=horizons)
    directions_df = compute_direction_labels(df, returns_df, horizons=horizons)

    combined_labels = pd.concat([returns_df, directions_df], axis=1)

    if include_timestamp:
        # Prepend alignment timestamps
        time_cols = [c for c in ["time", "timestamp"] if c in df.columns]
        if time_cols:
            result_df = pd.concat([df[time_cols], combined_labels], axis=1)
        else:
            result_df = combined_labels
    else:
        result_df = combined_labels

    return result_df, label_defs


def generate_label_metadata(
    df_labels: pd.DataFrame,
    label_defs: list[LabelDefinition],
    spread_series: pd.Series | None = None,
) -> dict[str, Any]:
    """Generate comprehensive machine-readable metadata for all target labels.

    Parameters
    ----------
    df_labels : pd.DataFrame
        DataFrame containing computed labels.
    label_defs : list[LabelDefinition]
        Catalog of label definitions.
    spread_series : pd.Series | None, default None
        Optional spread series for cost-relationship diagnostics.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary.
    """
    total_rows = len(df_labels)
    label_details: list[dict[str, Any]] = []

    mean_spread_pts = float(spread_series.mean()) if spread_series is not None else None

    for ld in label_defs:
        d = ld.to_dict()
        if ld.name in df_labels.columns:
            s = df_labels[ld.name]
            nan_cnt = int(s.isna().sum())
            clean_s = s.dropna()

            if ld.label_type in ("multiclass_direction", "volatility_scaled_direction"):
                counts = clean_s.value_counts().to_dict()
                total_valid = len(clean_s)
                long_cnt = int(counts.get(1.0, 0))
                neutral_cnt = int(counts.get(0.0, 0))
                short_cnt = int(counts.get(-1.0, 0))

                long_pct = float(long_cnt / total_valid * 100.0) if total_valid > 0 else 0.0
                neutral_pct = float(neutral_cnt / total_valid * 100.0) if total_valid > 0 else 0.0
                short_pct = float(short_cnt / total_valid * 100.0) if total_valid > 0 else 0.0

                min_class_cnt = min(long_cnt, short_cnt) if min(long_cnt, short_cnt) > 0 else 1
                max_class_cnt = max(long_cnt, neutral_cnt, short_cnt)
                imbalance_ratio = round(max_class_cnt / min_class_cnt, 2)

                d.update(
                    {
                        "total_valid": total_valid,
                        "nan_count": nan_cnt,
                        "class_counts": {
                            "LONG (+1)": long_cnt,
                            "NEUTRAL (0)": neutral_cnt,
                            "SHORT (-1)": short_cnt,
                        },
                        "class_percentages": {
                            "LONG (+1)": round(long_pct, 2),
                            "NEUTRAL (0)": round(neutral_pct, 2),
                            "SHORT (-1)": round(short_pct, 2),
                        },
                        "imbalance_ratio": imbalance_ratio,
                    }
                )
            else:
                q = clean_s.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_dict()
                quantiles = {f"p{int(k * 100):02d}": float(v) for k, v in q.items()}

                # Ratio of absolute return magnitude to average spread
                spread_ratio = None
                if mean_spread_pts is not None and "close" in df_labels.columns:
                    mean_close = float(df_labels["close"].mean())
                    mean_spread_pct = (mean_spread_pts * 1e-5) / mean_close
                    spread_ratio = float(clean_s.abs().mean() / mean_spread_pct)

                d.update(
                    {
                        "total_valid": len(clean_s),
                        "nan_count": nan_cnt,
                        "mean": float(clean_s.mean()),
                        "median": float(clean_s.median()),
                        "std": float(clean_s.std()),
                        "min": float(clean_s.min()),
                        "max": float(clean_s.max()),
                        "quantiles": quantiles,
                        "abs_movement_to_mean_spread_ratio": spread_ratio,
                    }
                )
        label_details.append(d)

    return {
        "total_labels": len(label_defs),
        "total_rows": total_rows,
        "candidate_horizons_bars": list(set(ld.horizon_bars for ld in label_defs)),
        "candidate_horizons_minutes": list(set(ld.horizon_minutes for ld in label_defs)),
        "labels": label_details,
    }
