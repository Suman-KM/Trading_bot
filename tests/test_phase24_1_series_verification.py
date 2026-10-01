"""Test suite for Phase 24.1 German 2Y Series Verification & Formal Approval.

Validates:
1. Original Phase 23 identifier (FRED: IRLTLT01DEM156N) is correctly classified as monthly 10-year.
2. Current Bundesbank identifier has verified official metadata.
3. Current Bundesbank series maturity is explicitly documented as 2.0 years.
4. Current series frequency is strictly daily (P1D).
5. Source identifier is reproducible and verified.
6. Approved series is the series actually referenced by Phase 24 processed data.
7. No unauthorized alternative series is silently substituted.
8. Existing causal alignment convention remains unchanged.
9. Phase 11 test partition boundary rejection is strictly enforced.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai.backtest.robustness import (
    PHASE11_TEST_LOCK_TIMESTAMP,
    VALIDATION_END_TIMESTAMP,
    verify_test_partition_rejection,
)
from ai.data.exogenous.yields import (
    compute_file_sha256,
    load_bundesbank_2y,
    load_fred_german_yield,
    validate_raw_series,
)

BUNDESBANK_APPROVED_KEY = "BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A"
FRED_ORIGINAL_KEY = "IRLTLT01DEM156N"


def test_phase11_test_lock_boundary_rejection():
    """Governance test: ensure Phase 11 test partition boundary is strictly rejected."""
    verify_test_partition_rejection(VALIDATION_END_TIMESTAMP)

    violating_time = PHASE11_TEST_LOCK_TIMESTAMP + timedelta(hours=4)
    with pytest.raises(ValueError, match="CRITICAL TEST PARTITION VIOLATION"):
        verify_test_partition_rejection(violating_time)


def test_original_phase23_identifier_classification():
    """1. Test that original Phase 23 identifier (IRLTLT01DEM156N) is classified as monthly 10Y."""
    raw_fred_path = Path("data/external/macro_yields/raw/fred_irltlt01dem156n.csv")
    assert raw_fred_path.exists(), "Raw FRED IRLTLT01DEM156N CSV must exist"

    df_fred = load_fred_german_yield(raw_fred_path)
    val = validate_raw_series(
        df_fred, "date", "irltlt01dem156n", FRED_ORIGINAL_KEY, expected_freq="monthly"
    )

    # Must have monthly frequency (gaps ~30 days)
    assert val.frequency_consistent is True
    # If audited as daily, it must be flagged as frequency inconsistent
    val_as_daily = validate_raw_series(
        df_fred, "date", "irltlt01dem156n", FRED_ORIGINAL_KEY, expected_freq="daily"
    )
    assert val_as_daily.frequency_consistent is False
    assert any("Frequency mismatch" in a for a in val_as_daily.anomalies)


def test_current_bundesbank_identifier_metadata():
    """2. Test that current Bundesbank identifier has verified official metadata."""
    raw_buba_path = Path("data/external/macro_yields/raw/bundesbank_2y.csv")
    assert raw_buba_path.exists(), "Raw Bundesbank CSV must exist"

    with open(raw_buba_path, "r", encoding="utf-8") as f:
        header_lines = [f.readline() for _ in range(8)]

    # Verify series key is in header
    assert BUNDESBANK_APPROVED_KEY in header_lines[0]
    # Verify official German title mentions "RLZ 2 Jahre" (residual maturity 2 years)
    assert "RLZ 2 Jahre" in header_lines[1] or "residual maturity of 2.0 years" in header_lines[1]
    # Verify daily time format code
    time_format_line = next(
        line for line in header_lines if "Format der Zeitangabe" in line or "Time format" in line
    )
    assert "P1D" in time_format_line


def test_current_bundesbank_series_maturity():
    """3. Test that current Bundesbank series maturity is explicitly documented as 2.0 years."""
    raw_buba_path = Path("data/external/macro_yields/raw/bundesbank_2y.csv")
    with open(raw_buba_path, "r", encoding="utf-8") as f:
        title_line = f.readlines()[1]

    # Must contain explicit 2-year maturity definition
    has_2y_de = "RLZ 2 Jahre" in title_line
    has_2y_en = "residual maturity of 2.0 years" in title_line
    assert has_2y_de or has_2y_en, f"Maturity not explicitly 2.0 years in title: {title_line}"


def test_current_series_frequency_is_daily():
    """4. Test that current Bundesbank series frequency is strictly daily."""
    raw_buba_path = Path("data/external/macro_yields/raw/bundesbank_2y.csv")
    df_buba = load_bundesbank_2y(raw_buba_path)
    val = validate_raw_series(
        df_buba, "date", "german_2y", BUNDESBANK_APPROVED_KEY, expected_freq="daily"
    )

    assert val.frequency_consistent is True
    assert val.row_count > 10000
    assert val.valid_numeric_count > 7000


def test_source_identifier_is_reproducible():
    """5. Test that source identifier is reproducible and verified with SHA-256."""
    prov_path = Path("data/external/macro_yields/metadata/yield_provenance.json")
    assert prov_path.exists(), "yield_provenance.json must exist"

    with open(prov_path, "r", encoding="utf-8") as f:
        prov = json.load(f)

    assert "bundesbank_2y" in prov
    buba_meta = prov["bundesbank_2y"]
    assert BUNDESBANK_APPROVED_KEY in buba_meta["source_identifier"]

    # Verify SHA-256 hash matches disk file
    actual_hash = compute_file_sha256(buba_meta["raw_file_path"])
    assert actual_hash == buba_meta["sha256_hash"]


def test_approved_series_is_referenced_by_phase24_processed_data():
    """6. Test that approved series is the series actually referenced by Phase 24 processed data."""
    processed_parquet = Path("data/external/macro_yields/processed/eurusd_h4_yield_aligned.parquet")
    assert processed_parquet.exists(), "Processed aligned parquet must exist"

    df = pd.read_parquet(processed_parquet)
    assert "German_2Y" in df.columns
    assert "US_Germany_2Y_Spread" in df.columns
    assert len(df) == 25800

    # Ensure spread equals US_2Y - German_2Y exactly
    diff = df["US_Germany_2Y_Spread"] - (df["US_2Y"] - df["German_2Y"])
    assert np.allclose(diff.values, 0.0)


def test_no_unauthorized_alternative_series_silently_substituted():
    """7. Test that no unauthorized alternative series is silently substituted."""
    prov_path = Path("data/external/macro_yields/metadata/yield_provenance.json")
    with open(prov_path, "r", encoding="utf-8") as f:
        prov = json.load(f)

    # Exactly 3 known tracked series: DGS2, IRLTLT01DEM156N (audited), BBSIS... (approved)
    expected_keys = {"fred_dgs2", "fred_irltlt01dem156n", "bundesbank_2y"}
    assert set(prov.keys()) == expected_keys

    # Verify Bundesbank series key is exactly the approved key
    assert prov["bundesbank_2y"]["source_identifier"] == f"Bundesbank: {BUNDESBANK_APPROVED_KEY}"


def test_existing_causal_alignment_convention_remains_unchanged():
    """8. Test that existing causal alignment convention (Day D+1 00:00 UTC) remains unchanged."""
    processed_parquet = Path("data/external/macro_yields/processed/eurusd_h4_yield_aligned.parquet")
    df = pd.read_parquet(processed_parquet)

    # Verify invariant: data_availability_timestamp <= timestamp
    avail_ts = pd.to_datetime(df["data_availability_timestamp"], utc=True)
    bar_ts = pd.to_datetime(df["timestamp"], utc=True)
    assert (avail_ts <= bar_ts).all(), "Found lookahead violation in aligned dataset"

    # Verify zero same-day leakage
    obs_date = pd.to_datetime(df["yield_observation_date"]).dt.date
    bar_date = bar_ts.dt.date
    assert not (obs_date == bar_date).any(), "Found same-day yield leak in aligned dataset"
