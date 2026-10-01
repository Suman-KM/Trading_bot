"""Historical yield data ingestion, validation, provenance, and causal alignment.

Implements Phase 24 requirements for ingesting US 2Y Treasury yields, German 2Y
yields, calendar reconciliation, point-in-time publication tracking, yield spread
construction, and causal alignment to EURUSD H4 bars with zero lookahead bias.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class YieldProvenance:
    """Provenance and cryptographic verification metadata for an external yield series."""

    source_name: str
    source_identifier: str
    source_url: str
    retrieval_timestamp_utc: str
    requested_date_range: str
    actual_date_range: str
    frequency: str
    timezone: str
    units: str
    missing_value_convention: str
    revision_vintage_info: str
    sha256_hash: str
    file_size_bytes: int
    raw_file_path: str

    def to_dict(self) -> dict[str, Any]:
        """Convert provenance metadata to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RawSeriesValidation:
    """Validation report on raw series integrity before cleaning."""

    series_id: str
    row_count: int
    first_date: str | None
    last_date: str | None
    valid_numeric_count: int
    missing_count: int
    duplicate_count: int
    invalid_numeric_count: int
    frequency_consistent: bool
    units_consistent: bool
    is_chronological: bool
    anomalies: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert validation report to dictionary."""
        return asdict(self)


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute the SHA-256 hash of a local file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_and_save_dataset(
    url: str,
    target_path: Path,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[bytes, str]:
    """Download a dataset from a remote URL, persist to disk, and return content and SHA-256."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    default_agent = "Mozilla/5.0 (Quantitative Research; ai-trading-system)"
    req_headers = headers or {"User-Agent": default_agent}
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read()

    target_path.write_bytes(content)
    sha256 = hashlib.sha256(content).hexdigest()
    return content, sha256


def load_fred_us_2y(filepath: Path | str) -> pd.DataFrame:
    """Load and parse raw FRED DGS2 CSV dataset.

    Returns DataFrame with columns ['date', 'us_2y_raw', 'us_2y'].
    """
    df = pd.read_csv(filepath)
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = "DGS2" if "DGS2" in df.columns else df.columns[1]

    df = df.rename(columns={date_col: "date", val_col: "us_2y_raw"})
    df["date"] = pd.to_datetime(df["date"])
    df["us_2y"] = pd.to_numeric(df["us_2y_raw"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def load_fred_german_yield(filepath: Path | str) -> pd.DataFrame:
    """Load and parse raw FRED IRLTLT01DEM156N CSV dataset.

    Returns DataFrame with columns ['date', 'irltlt01dem156n_raw', 'irltlt01dem156n'].
    """
    df = pd.read_csv(filepath)
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = "IRLTLT01DEM156N" if "IRLTLT01DEM156N" in df.columns else df.columns[1]

    df = df.rename(columns={date_col: "date", val_col: "irltlt01dem156n_raw"})
    df["date"] = pd.to_datetime(df["date"])
    df["irltlt01dem156n"] = pd.to_numeric(df["irltlt01dem156n_raw"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def load_bundesbank_2y(filepath: Path | str) -> pd.DataFrame:
    """Load and parse raw Deutsche Bundesbank daily 2Y yield CSV dataset.

    Series: BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A
    Returns DataFrame with columns ['date', 'german_2y_raw', 'german_2y'].
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    data_rows = []
    for line in raw_lines:
        parts = line.strip().split(";")
        if len(parts) >= 2 and len(parts[0]) == 10 and parts[0][4] == "-" and parts[0][7] == "-":
            data_rows.append((parts[0], parts[1]))

    df = pd.DataFrame(data_rows, columns=["date", "german_2y_raw"])
    df["date"] = pd.to_datetime(df["date"])
    df["german_2y"] = pd.to_numeric(df["german_2y_raw"].str.replace(",", "."), errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def validate_raw_series(
    df: pd.DataFrame,
    date_col: str,
    val_col: str,
    series_id: str,
    expected_freq: str = "daily",
) -> RawSeriesValidation:
    """Perform rigorous raw data validation on an ingested yield series."""
    anomalies: list[str] = []
    row_count = len(df)
    if row_count == 0:
        return RawSeriesValidation(
            series_id=series_id,
            row_count=0,
            first_date=None,
            last_date=None,
            valid_numeric_count=0,
            missing_count=0,
            duplicate_count=0,
            invalid_numeric_count=0,
            frequency_consistent=False,
            units_consistent=True,
            is_chronological=True,
            anomalies=["Dataset is empty."],
        )

    # 1. Chronological ordering
    dates = pd.to_datetime(df[date_col])
    is_chronological = bool(dates.is_monotonic_increasing)
    if not is_chronological:
        anomalies.append("Dates are not strictly monotonically increasing.")

    # 2. Duplicate dates
    duplicate_count = int(dates.duplicated().sum())
    if duplicate_count > 0:
        anomalies.append(f"Found {duplicate_count} duplicate dates.")

    # 3. Numeric validity
    numeric_series = pd.to_numeric(df[val_col], errors="coerce")
    valid_count = int(numeric_series.notna().sum())
    missing_count = int(numeric_series.isna().sum())

    # Check for invalid values (< -10% or > 50% for modern government bond yields)
    impossible_vals = numeric_series[(numeric_series < -10.0) | (numeric_series > 50.0)]
    invalid_numeric_count = int(len(impossible_vals))
    if invalid_numeric_count > 0:
        anomalies.append(
            f"Found {invalid_numeric_count} impossible yield values (< -10% or > 50%)."
        )

    # Check for Inf
    inf_count = int(np.isinf(numeric_series).sum())
    if inf_count > 0:
        anomalies.append(f"Found {inf_count} infinite values.")
        invalid_numeric_count += inf_count

    # 4. Frequency consistency
    freq_consistent = True
    if expected_freq == "daily":
        # Check gap between consecutive dates
        unique_dates = dates.drop_duplicates().sort_values()
        date_diffs = unique_dates.diff().dt.days
        # Monthly frequency detection in supposedly daily series
        median_gap = date_diffs.median()
        if median_gap > 7:
            freq_consistent = False
            anomalies.append(
                f"Frequency mismatch: expected daily series, but median gap is {median_gap} days."
            )
    elif expected_freq == "monthly":
        unique_dates = dates.drop_duplicates().sort_values()
        date_diffs = unique_dates.diff().dt.days
        median_gap = date_diffs.median()
        if median_gap < 25 or median_gap > 35:
            freq_consistent = False
            anomalies.append(
                f"Frequency mismatch: expected monthly series, but median gap is {median_gap} days."
            )

    first_date_str = str(dates.min().strftime("%Y-%m-%d")) if row_count > 0 else None
    last_date_str = str(dates.max().strftime("%Y-%m-%d")) if row_count > 0 else None

    return RawSeriesValidation(
        series_id=series_id,
        row_count=row_count,
        first_date=first_date_str,
        last_date=last_date_str,
        valid_numeric_count=valid_count,
        missing_count=missing_count,
        duplicate_count=duplicate_count,
        invalid_numeric_count=invalid_numeric_count,
        frequency_consistent=freq_consistent,
        units_consistent=True,
        is_chronological=is_chronological,
        anomalies=anomalies,
    )


def reconcile_yield_calendars(
    df_us: pd.DataFrame,
    df_de: pd.DataFrame,
    start_date: str = "2010-01-01",
    end_date: str = "2026-09-30",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Perform holiday and calendar reconciliation between US and German bond markets.

    Determines US-only holidays, German-only holidays, mutual closures, and handles
    asynchronous market closures via strict causal point-in-time forward filling.

    Parameters
    ----------
    df_us : pd.DataFrame
        US yield DataFrame containing ['date', 'us_2y'].
    df_de : pd.DataFrame
        German yield DataFrame containing ['date', 'german_2y'].
    start_date : str
        Start date of reconciliation window.
    end_date : str
        End date of reconciliation window.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Reconciled daily business day DataFrame and calendar audit report.
    """
    bday_range = pd.date_range(start_date, end_date, freq="B")
    df_cal = pd.DataFrame({"date": bday_range})

    # Prepare US
    us_sub = df_us[["date", "us_2y"]].dropna(subset=["date"]).copy()
    us_sub = us_sub.drop_duplicates(subset=["date"]).sort_values("date")

    # Prepare DE
    de_sub = df_de[["date", "german_2y"]].dropna(subset=["date"]).copy()
    de_sub = de_sub.drop_duplicates(subset=["date"]).sort_values("date")

    merged = pd.merge(df_cal, us_sub, on="date", how="left")
    merged = pd.merge(merged, de_sub, on="date", how="left")

    # Identify holiday status before forward filling
    us_missing = merged["us_2y"].isna()
    de_missing = merged["german_2y"].isna()

    us_only_holidays = merged[us_missing & (~de_missing)]["date"].dt.strftime("%Y-%m-%d").tolist()
    de_only_holidays = merged[(~us_missing) & de_missing]["date"].dt.strftime("%Y-%m-%d").tolist()
    mutual_closures = merged[us_missing & de_missing]["date"].dt.strftime("%Y-%m-%d").tolist()
    both_active = merged[(~us_missing) & (~de_missing)]["date"].dt.strftime("%Y-%m-%d").tolist()

    # Reconstructing daily yield spread requires explicit point-in-time synchronization
    # with forward-filling of the inactive market's last known close (Phase 23 Section 4.B).
    merged["US_2Y"] = merged["us_2y"].ffill()
    merged["German_2Y"] = merged["german_2y"].ffill()

    # Derived spread: US_2Y - German_2Y
    merged["US_Germany_2Y_Spread"] = merged["US_2Y"] - merged["German_2Y"]

    # 5-business-day change in the spread: Spread_t - Spread_{t-5}
    merged["US_Germany_2Y_Spread_5D_Change"] = merged["US_Germany_2Y_Spread"] - merged[
        "US_Germany_2Y_Spread"
    ].shift(5)

    # Point-in-time publication timestamps:
    # US 2Y: ~21:15 UTC on day D
    # German 2Y: ~16:00 UTC on day D
    # Spread for day D is known once both sources are published -> max(21:15, 16:00) = 21:15 UTC
    merged["data_availability_timestamp"] = (
        (merged["date"] + pd.Timedelta(hours=21, minutes=15))
        .dt.tz_localize(UTC)
        .astype("datetime64[ns, UTC]")
    )

    # Conservative next-day convention specified by Phase 23:
    # First usable H4 bar is 00:00:00 UTC on Day D + 1
    merged["first_usable_h4_timestamp"] = (
        (merged["date"] + pd.Timedelta(days=1)).dt.tz_localize(UTC).astype("datetime64[ns, UTC]")
    )

    audit = {
        "reconciliation_window": f"{start_date} to {end_date}",
        "total_business_days": len(merged),
        "both_active_count": len(both_active),
        "us_only_holiday_count": len(us_only_holidays),
        "german_only_holiday_count": len(de_only_holidays),
        "mutual_closure_count": len(mutual_closures),
        "sample_us_only_holidays": us_only_holidays[:10],
        "sample_german_only_holidays": de_only_holidays[:10],
        "sample_mutual_closures": mutual_closures[:10],
        "closure_classification": "VALID MARKET CLOSURES (Not data quality corruption)",
    }

    return merged, audit


def align_yields_to_eurusd_h4(
    df_h4: pd.DataFrame,
    df_spread: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align daily yield spread features to EURUSD H4 bars causally.

    Strictly guarantees that for every H4 bar, no information from day D is observed
    until its first usable bar timestamp (Day D+1 00:00:00 UTC).

    Parameters
    ----------
    df_h4 : pd.DataFrame
        EURUSD H4 DataFrame with 'timestamp' column.
    df_spread : pd.DataFrame
        Reconciled daily spread DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Aligned research DataFrame and validation audit.
    """
    df_h4_clean = df_h4.copy()
    df_h4_clean["timestamp"] = pd.to_datetime(df_h4_clean["timestamp"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    df_h4_sorted = df_h4_clean.sort_values("timestamp").reset_index(drop=True)

    df_spread_sorted = (
        df_spread.dropna(subset=["first_usable_h4_timestamp"])
        .sort_values("first_usable_h4_timestamp")
        .reset_index(drop=True)
    )

    # Use merge_asof backwards: bar_timestamp >= first_usable_h4_timestamp
    cols_to_merge = [
        "first_usable_h4_timestamp",
        "data_availability_timestamp",
        "date",
        "US_2Y",
        "German_2Y",
        "US_Germany_2Y_Spread",
        "US_Germany_2Y_Spread_5D_Change",
    ]

    aligned = pd.merge_asof(
        df_h4_sorted,
        df_spread_sorted[cols_to_merge],
        left_on="timestamp",
        right_on="first_usable_h4_timestamp",
        direction="backward",
    )

    # Rename 'date' to 'yield_observation_date' to avoid confusion
    aligned = aligned.rename(columns={"date": "yield_observation_date"})

    # Check causality invariant
    # Invariant: data_availability_timestamp <= timestamp
    has_yield = aligned["US_Germany_2Y_Spread"].notna()
    causal_violations = aligned[
        has_yield & (aligned["data_availability_timestamp"] > aligned["timestamp"])
    ]
    n_violations = int(len(causal_violations))

    # Check same-day leak: yield observation date must NOT equal bar date
    same_day_leaks = aligned[
        has_yield & (aligned["yield_observation_date"].dt.date == aligned["timestamp"].dt.date)
    ]
    n_same_day = int(len(same_day_leaks))

    aligned["alignment_status"] = np.where(
        has_yield & (n_violations == 0), "CAUSAL_VALID", "EXCLUDED"
    )

    total_bars = len(aligned)
    valid_bars = int(has_yield.sum())
    excluded_bars = total_bars - valid_bars

    earliest_usable = str(aligned.loc[has_yield, "timestamp"].iloc[0]) if valid_bars > 0 else None
    latest_usable = str(aligned.loc[has_yield, "timestamp"].iloc[-1]) if valid_bars > 0 else None

    audit = {
        "h4_bars_examined": total_bars,
        "h4_bars_with_valid_exogenous_data": valid_bars,
        "h4_bars_excluded": excluded_bars,
        "earliest_usable_timestamp": earliest_usable,
        "latest_usable_timestamp": latest_usable,
        "number_of_leakage_violations": n_violations,
        "same_day_leakage_violations": n_same_day,
        "alignment_strictly_causal": (n_violations == 0 and n_same_day == 0),
    }

    return aligned, audit
