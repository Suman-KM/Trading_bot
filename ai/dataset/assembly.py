"""Deterministic dataset assembly engine.

Joins point-in-time features (Phase 5) and future labels (Phase 6) on UTC timestamps,
enforcing strict causal separation and eliminating feature warm-up NaNs while preserving
forward-horizon label NaNs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ai.features.pipeline import get_feature_registry
from ai.labels.returns import DEFAULT_HORIZONS

DEFAULT_FEATURES_PATH = Path("data/features/eurusd_m15/eurusd_m15_features.parquet")
DEFAULT_LABELS_PATH = Path("data/labels/eurusd_m15/eurusd_m15_labels.parquet")


class DataLeakageError(ValueError):
    """Raised when data leakage between features, targets, or splits is detected."""


@dataclass
class AssembledDataset:
    """Container for joined features, labels, and target definitions.

    Attributes
    ----------
    df : pd.DataFrame
        Complete joined dataframe (post feature-warmup removal).
    feature_names : list[str]
        List of strictly causal input feature column names (X).
    target_name : str
        Selected target label column name (y).
    horizon_bars : int
        Lookahead prediction horizon in bars.
    timestamp_col : str
        Name of the primary timestamp column.
    """

    df: pd.DataFrame
    feature_names: list[str]
    target_name: str
    horizon_bars: int
    timestamp_col: str = "timestamp"

    @property
    def X(self) -> pd.DataFrame:
        """Point-in-time input feature matrix X_t."""
        return self.df[self.feature_names]

    @property
    def y(self) -> pd.Series:
        """Forward-looking target label vector y_t."""
        return self.df[self.target_name]

    @property
    def timestamps(self) -> pd.Series:
        """Timestamp series aligned with observations."""
        return self.df[self.timestamp_col]

    @property
    def usable_mask(self) -> pd.Series:
        """Boolean mask indicating complete rows (valid features AND valid target)."""
        valid_target = ~self.df[self.target_name].isna()
        valid_features = ~self.df[self.feature_names].isna().any(axis=1)
        return valid_target & valid_features

    @property
    def usable_df(self) -> pd.DataFrame:
        """Dataframe containing strictly usable rows with complete features and non-NaN target."""
        return self.df.loc[self.usable_mask].copy()

    def get_usable(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Return (X, y, timestamps) filtered strictly to complete, non-NaN rows."""
        mask = self.usable_mask
        return (
            self.df.loc[mask, self.feature_names].copy(),
            self.df.loc[mask, self.target_name].copy(),
            self.df.loc[mask, self.timestamp_col].copy(),
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert dataset summary to metadata dictionary."""
        mask = self.usable_mask
        return {
            "total_assembled_rows": len(self.df),
            "usable_rows": int(mask.sum()),
            "tail_label_nans": int((~mask).sum()),
            "feature_count": len(self.feature_names),
            "target_name": self.target_name,
            "horizon_bars": self.horizon_bars,
            "first_timestamp": str(self.df[self.timestamp_col].iloc[0]),
            "last_timestamp": str(self.df[self.timestamp_col].iloc[-1]),
        }


def infer_horizon_from_target(target_name: str) -> int:
    """Infer the horizon in bars from standard label naming conventions.

    Parameters
    ----------
    target_name : str
        Target column name (e.g. 'direction_4', 'future_return_16', 'direction_vol_1').

    Returns
    -------
    int
        Lookahead horizon in bars.

    Raises
    ------
    ValueError
        If the horizon cannot be determined from the target name.
    """
    for h in sorted(DEFAULT_HORIZONS, reverse=True):
        if target_name.endswith(f"_{h}"):
            return h
    raise ValueError(
        f"Cannot infer horizon from target '{target_name}'. "
        f"Please specify horizon_bars explicitly (supported: {DEFAULT_HORIZONS})."
    )


def is_label_or_future_column(col_name: str) -> bool:
    """Check if a column name represents a future-looking label or target."""
    if col_name.startswith("future_") or col_name.startswith("direction_"):
        return True
    return False


def assemble_dataset(
    features_df: pd.DataFrame | None = None,
    labels_df: pd.DataFrame | None = None,
    features_path: Path | str | None = None,
    labels_path: Path | str | None = None,
    target_column: str = "direction_4",
    horizon_bars: int | None = None,
    feature_columns: list[str] | None = None,
    drop_feature_warmup: bool = True,
    timestamp_col: str = "timestamp",
) -> AssembledDataset:
    """Assemble a unified, leakage-safe dataset from feature and label sources.

    Parameters
    ----------
    features_df : pd.DataFrame | None, optional
        Pre-loaded feature dataframe.
    labels_df : pd.DataFrame | None, optional
        Pre-loaded label dataframe.
    features_path : Path | str | None, optional
        Path to features parquet file. Defaults to data/features/eurusd_m15/...
    labels_path : Path | str | None, optional
        Path to labels parquet file. Defaults to data/labels/eurusd_m15/...
    target_column : str, default 'direction_4'
        Target label column name.
    horizon_bars : int | None, optional
        Lookahead prediction horizon in bars. If None, inferred from target_column.
    feature_columns : list[str] | None, optional
        Specific feature column names to include in X. Defaults to all 80 derived features.
    drop_feature_warmup : bool, default True
        If True, drops the initial rows (e.g. first 80) where features have warm-up NaNs.
    timestamp_col : str, default 'timestamp'
        Name of timestamp column to use for joins and temporal sorting.

    Returns
    -------
    AssembledDataset
        The assembled and validated dataset container.

    Raises
    ------
    FileNotFoundError
        If parquet files are missing.
    ValueError
        If inputs or columns are invalid.
    DataLeakageError
        If label or future-looking columns are detected in feature_columns.
    """
    # 1. Load dataframes if not provided
    if features_df is None:
        p_feat = Path(features_path) if features_path else DEFAULT_FEATURES_PATH
        if not p_feat.exists():
            raise FileNotFoundError(f"Features file not found at: {p_feat}")
        features_df = pd.read_parquet(p_feat)

    if labels_df is None:
        p_lab = Path(labels_path) if labels_path else DEFAULT_LABELS_PATH
        if not p_lab.exists():
            raise FileNotFoundError(f"Labels file not found at: {p_lab}")
        labels_df = pd.read_parquet(p_lab)

    # 2. Validate timestamp column
    if timestamp_col not in features_df.columns:
        raise ValueError(f"Features dataframe missing timestamp column: '{timestamp_col}'")
    if timestamp_col not in labels_df.columns:
        raise ValueError(f"Labels dataframe missing timestamp column: '{timestamp_col}'")

    # 3. Resolve horizon
    if horizon_bars is None:
        horizon_bars = infer_horizon_from_target(target_column)
    elif target_column.endswith(f"_{horizon_bars}") is False and any(
        target_column.endswith(f"_{h}") for h in DEFAULT_HORIZONS
    ):
        inferred = infer_horizon_from_target(target_column)
        if inferred != horizon_bars:
            raise ValueError(
                f"Specified horizon_bars={horizon_bars} conflicts with "
                f"target '{target_column}' (inferred horizon: {inferred})."
            )

    # 4. Resolve and validate feature columns
    if feature_columns is None:
        # Default to all 80 derived quantitative features
        registry = get_feature_registry()
        feature_columns = [f.name for f in registry]

    # Verify no label or target is in feature_columns (CRITICAL LEAKAGE CHECK)
    for col in feature_columns:
        if is_label_or_future_column(col) or col == target_column or col in labels_df.columns:
            if col not in ["time", "timestamp"]:
                raise DataLeakageError(
                    f"CRITICAL LEAKAGE ERROR: Target/label column '{col}' is present "
                    "in feature_columns! Future information must never leak into X."
                )

    missing_features = [c for c in feature_columns if c not in features_df.columns]
    if missing_features:
        raise ValueError(f"Requested features missing from feature dataset: {missing_features}")

    if target_column not in labels_df.columns:
        raise ValueError(
            f"Target column '{target_column}' missing from label dataset. "
            f"Available targets: {[c for c in labels_df.columns if c not in ['time', 'timestamp']]}"
        )

    # 5. Check timestamp ordering and uniqueness
    if not features_df[timestamp_col].is_monotonic_increasing:
        raise ValueError("Features dataframe timestamps are not monotonically increasing.")
    if not labels_df[timestamp_col].is_monotonic_increasing:
        raise ValueError("Labels dataframe timestamps are not monotonically increasing.")

    if features_df[timestamp_col].duplicated().any():
        raise ValueError("Features dataframe contains duplicate timestamps.")
    if labels_df[timestamp_col].duplicated().any():
        raise ValueError("Labels dataframe contains duplicate timestamps.")

    # 6. Join features and target
    # Select columns to retain
    cols_to_keep_feat = [timestamp_col]
    if "time" in features_df.columns and "time" not in cols_to_keep_feat:
        cols_to_keep_feat.append("time")
    cols_to_keep_feat.extend([c for c in feature_columns if c not in cols_to_keep_feat])

    cols_to_keep_lab = [timestamp_col, target_column]

    feat_sub = features_df[cols_to_keep_feat]
    lab_sub = labels_df[cols_to_keep_lab]

    # Deterministic inner join
    joined = pd.merge(feat_sub, lab_sub, on=timestamp_col, how="inner")

    if len(joined) == 0:
        raise ValueError("Timestamp intersection between features and labels is empty.")

    # Ensure strictly chronological
    joined = joined.sort_values(timestamp_col).reset_index(drop=True)

    # 7. Warm-up handling
    if drop_feature_warmup:
        # Exclude rows without complete required features
        nan_mask = joined[feature_columns].isna().any(axis=1)
        joined = joined.loc[~nan_mask].reset_index(drop=True)

        # Verify zero NaNs in features
        remaining_feat_nans = joined[feature_columns].isna().sum().sum()
        if remaining_feat_nans > 0:
            raise ValueError(
                f"Feature matrix still contains {remaining_feat_nans} NaNs after warm-up drop."
            )

    # 8. Preserving label NaNs where horizon requires future data
    # Note: joined retains rows where target is NaN (e.g. the final H rows)
    return AssembledDataset(
        df=joined,
        feature_names=feature_columns,
        target_name=target_column,
        horizon_bars=horizon_bars,
        timestamp_col=timestamp_col,
    )
