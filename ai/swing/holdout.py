"""Phase 18: Fresh Research Holdout Governance and State Machine.

Enforces strict chronological holdout governance before feature experiments:
1. Pre-Holdout Research Partition: 2010-03-01 16:00 UTC to 2024-11-04 12:00 UTC
2. Purge Gap: 8 H4 bars (32 hours)
3. Fresh Research Holdout Partition: 2024-11-06 00:00 UTC to 2026-02-19 10:45 UTC
4. Locked Test Partition: 2026-02-19 12:00 UTC onward (PERMANENTLY LOCKED)

The Fresh Research Holdout is sealed and raises PermissionError if accessed
prior to explicit feature, model, and protocol freezing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

PRE_HOLDOUT_RESEARCH_START_TS = pd.Timestamp("2010-03-01 16:00:00+00:00")
PRE_HOLDOUT_RESEARCH_END_TS = pd.Timestamp("2024-11-04 12:00:00+00:00")
PURGE_GAP_BARS = 8
HOLDOUT_START_TS = pd.Timestamp("2024-11-06 00:00:00+00:00")
HOLDOUT_END_TS = pd.Timestamp("2026-02-19 10:45:00+00:00")
LOCKED_TEST_START_TS = pd.Timestamp("2026-02-19 12:00:00+00:00")


class ResearchHoldoutManager:
    """Manages the Fresh Phase 18 Research Holdout state and access permissions."""

    def __init__(self) -> None:
        self.is_features_frozen: bool = False
        self.is_model_frozen: bool = False
        self.is_protocol_frozen: bool = False
        self.features_frozen_at: str | None = None
        self.model_frozen_at: str | None = None
        self.protocol_frozen_at: str | None = None
        self.holdout_unlocked_at: str | None = None

    @property
    def is_sealed(self) -> bool:
        """True if holdout is still sealed and locked."""
        return self.holdout_unlocked_at is None

    @property
    def is_holdout_unlocked(self) -> bool:
        """True if holdout has been successfully unlocked."""
        return self.holdout_unlocked_at is not None

    def filter_pre_test_research(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to pre-test research partition, strictly rejecting locked test data."""
        ts = pd.to_datetime(df["timestamp"], utc=True)
        if (ts >= LOCKED_TEST_START_TS).any():
            raise PermissionError("Attempted to access locked test partition data.")
        return df[ts < LOCKED_TEST_START_TS].copy().reset_index(drop=True)

    def freeze_features(self, timestamp_iso: str) -> None:
        """Declare that candidate feature definitions and selections are permanently frozen."""
        self.is_features_frozen = True
        self.features_frozen_at = timestamp_iso

    def freeze_model(self, timestamp_iso: str) -> None:
        """Declare that model family and hyperparameters are permanently frozen."""
        self.is_model_frozen = True
        self.model_frozen_at = timestamp_iso

    def freeze_protocol(self, timestamp_iso: str) -> None:
        """Declare that evaluation protocol and threshold logic are permanently frozen."""
        self.is_protocol_frozen = True
        self.protocol_frozen_at = timestamp_iso

    def unlock_holdout(self, timestamp_iso: str = "") -> None:
        """Unlock the Fresh Research Holdout for single out-of-sample evaluation.

        Raises PermissionError if features, model, or protocol are not yet frozen.
        """
        if not timestamp_iso:
            from datetime import datetime, timezone

            timestamp_iso = datetime.now(timezone.utc).isoformat()
        if not (self.is_features_frozen and self.is_model_frozen and self.is_protocol_frozen):
            raise PermissionError(
                "Cannot unlock Fresh Research Holdout before freezing features, "
                "model, and protocol!"
            )
        self.holdout_unlocked_at = timestamp_iso

    def get_pre_holdout_research_data(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Retrieve the pre-holdout research partition (2010 to late 2024)."""
        mask = (df["timestamp"] >= PRE_HOLDOUT_RESEARCH_START_TS) & (
            df["timestamp"] <= PRE_HOLDOUT_RESEARCH_END_TS
        )
        return df[mask].copy().reset_index(drop=True)

    def get_holdout_data(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Retrieve the fresh research holdout partition (Nov 2024 to Feb 2026).

        Raises PermissionError if the holdout has not been unlocked.
        """
        if self.holdout_unlocked_at is None:
            raise PermissionError(
                "Fresh Research Holdout is SEALED. Access denied until frozen and unlocked."
            )
        mask = (df["timestamp"] >= HOLDOUT_START_TS) & (df["timestamp"] <= HOLDOUT_END_TS)
        return df[mask].copy().reset_index(drop=True)

    def get_status_dict(self) -> dict[str, Any]:
        """Return the current holdout governance audit metadata."""
        return {
            "pre_holdout_start": PRE_HOLDOUT_RESEARCH_START_TS.isoformat(),
            "pre_holdout_end": PRE_HOLDOUT_RESEARCH_END_TS.isoformat(),
            "purge_gap_bars": PURGE_GAP_BARS,
            "holdout_start": HOLDOUT_START_TS.isoformat(),
            "holdout_end": HOLDOUT_END_TS.isoformat(),
            "locked_test_start": LOCKED_TEST_START_TS.isoformat(),
            "is_features_frozen": self.is_features_frozen,
            "is_model_frozen": self.is_model_frozen,
            "is_protocol_frozen": self.is_protocol_frozen,
            "features_frozen_at": self.features_frozen_at,
            "model_frozen_at": self.model_frozen_at,
            "protocol_frozen_at": self.protocol_frozen_at,
            "holdout_unlocked_at": self.holdout_unlocked_at,
        }


# Alias for backward and forward compatibility
FreshResearchHoldoutManager = ResearchHoldoutManager
