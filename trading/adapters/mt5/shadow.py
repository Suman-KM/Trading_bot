"""Phase 40.4: Live Frozen Strategy Integration & Shadow Validation Pipeline.

Provides live M15 candle buffering, canonical point-in-time 80-feature computation,
frozen Random Forest model inference, directional confidence evaluation,
and non-executing shadow-mode signal validation under strict safety invariants.

ABSOLUTE GOVERNANCE & SAFETY INVARIANTS:
1. SHADOW MODE ONLY. execution_enabled is permanently False.
2. ZERO order_send() calls, zero demo orders, zero live orders.
3. Model is FROZEN: RandomForestBaseline (balanced, depth 10, leaf 20, estimators 100, seed 42).
4. Threshold is FROZEN: 0.60 directional confidence.
5. Features are FROZEN: exactly 80 canonical Phase 5 features.
6. Only completed M15 candles are evaluated (zero lookahead, zero incomplete candle leakage).
7. Warmup required: minimum 80 completed M15 bars.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import SplitConfig, split_dataset
from ai.features.pipeline import build_feature_pipeline
from ai.models.baselines import RandomForestBaseline
from trading.adapters.mt5.client import MT5ReadOnlyClient
from trading.adapters.mt5.safety import (
    ACCOUNT_TRADE_MODE_DEMO,
    DEMO_ONLY,
    LiveAccountForbiddenError,
)
from trading.adapters.mt5.schemas import MT5BarData, MT5Timeframe
from trading.execution.exceptions import BrokerExecutionDisabledError, MT5ConnectionError

logger = logging.getLogger(__name__)

M15_TIMEFRAME_MINUTES = 15
CANONICAL_FEATURE_COUNT = 80
MIN_WARMUP_BARS = 80
RECOMMENDED_WARMUP_BARS = 120
FROZEN_CONFIDENCE_THRESHOLD = 0.60


class FeatureVectorValidationError(ValueError):
    """Raised when an extracted feature vector violates canonical schema or integrity."""

    pass


class CandleIngestionError(ValueError):
    """Raised when candle timestamp or OHLC validation fails."""

    pass


class MarketClosedError(RuntimeError):
    """Raised when market is closed or trading on symbol is disabled."""

    pass


class StaleDataError(RuntimeError):
    """Raised when tick or bar data exceeds maximum staleness cutoff."""

    pass


class PreflightSafetyError(RuntimeError):
    """Raised when preflight safety gate fails."""

    pass


@dataclass
class ShadowPredictionRecord:
    """Detailed telemetry record for an individual frozen model evaluation."""

    timestamp_utc: str
    symbol: str
    timeframe: str
    candle_close_time_utc: str
    feature_count: int
    predicted_class: float
    predicted_class_name: str
    probability_short: float
    probability_neutral: float
    probability_long: float
    directional_confidence: float
    threshold: float
    signal_emitted: str
    shadow_signal: Optional[Dict[str, Any]] = None
    reason: str = "CONFIDENCE_BELOW_THRESHOLD"


@dataclass
class ShadowDiagnosticsMetrics:
    """Observability counters tracking live shadow evaluation."""

    m15_bars_loaded: int = 0
    m15_bars_completed_during_run: int = 0
    duplicate_bars_prevented: int = 0
    invalid_bars_rejected: int = 0
    model_evaluations: int = 0
    predictions: int = 0
    long_predictions: int = 0
    short_predictions: int = 0
    neutral_predictions: int = 0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    mean_confidence: float = 0.0
    median_confidence: float = 0.0
    count_confidence_ge_0_60: int = 0
    count_confidence_lt_0_60: int = 0
    signals_emitted: int = 0
    long_signals: int = 0
    short_signals: int = 0
    signals_rejected_by_threshold: int = 0
    feature_vector_failures: int = 0
    model_evaluation_failures: int = 0
    stale_data_events: int = 0
    orders_submitted: int = 0
    fills: int = 0
    open_positions: int = 0
    real_money_orders: int = 0


class M15CandleBuffer:
    """Deterministic, point-in-time buffer of completed M15 bars."""

    def __init__(self, max_capacity: int = 250) -> None:
        self.max_capacity = max_capacity
        self._bars: List[MT5BarData] = []
        self._seen_timestamps: set[datetime] = set()

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def bars(self) -> List[MT5BarData]:
        return list(self._bars)

    @property
    def latest_timestamp(self) -> Optional[datetime]:
        return self._bars[-1].timestamp_utc if self._bars else None

    def add_candle(self, bar: MT5BarData, current_time_utc: datetime) -> bool:
        """Add a single bar if and only if it is strictly completed and valid.

        Returns True if bar was appended, False if skipped (already seen or incomplete).
        """
        # 1. Incomplete Candle Guard: Bar must have closed at or before current time
        bar_close_time = bar.timestamp_utc + timedelta(minutes=M15_TIMEFRAME_MINUTES)
        if bar_close_time > current_time_utc:
            # Bar is still forming; reject to prevent lookahead leakage
            return False

        # 2. Duplicate Guard
        if bar.timestamp_utc in self._seen_timestamps:
            return False

        # 3. Monotonicity Guard
        if self._bars and bar.timestamp_utc <= self._bars[-1].timestamp_utc:
            raise CandleIngestionError(
                f"Timestamp regression: {bar.timestamp_utc} <= {self._bars[-1].timestamp_utc}"
            )

        # 4. OHLC Sanity
        if not (bar.low <= bar.open <= bar.high and bar.low <= bar.close <= bar.high):
            raise CandleIngestionError(f"Invalid OHLC structure on bar {bar.timestamp_utc}")

        self._bars.append(bar)
        self._seen_timestamps.add(bar.timestamp_utc)

        # Trim buffer to max capacity while maintaining contiguous order
        if len(self._bars) > self.max_capacity:
            discarded = self._bars.pop(0)
            self._seen_timestamps.discard(discarded.timestamp_utc)

        return True

    def populate_from_history(
        self, historical_bars: List[MT5BarData], current_time_utc: datetime
    ) -> int:
        """Populate buffer from historical bars, filtering strictly completed bars."""
        added = 0
        for b in historical_bars:
            if self.add_candle(b, current_time_utc=current_time_utc):
                added += 1
        return added

    def to_dataframe(self) -> pd.DataFrame:
        """Convert buffered bars into DataFrame expected by canonical feature pipeline."""
        if not self._bars:
            return pd.DataFrame()
        records = [
            {
                "timestamp": b.timestamp_utc,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "tick_volume": float(b.tick_volume),
                "spread": float(b.spread),
            }
            for b in self._bars
        ]
        return pd.DataFrame(records)


class LiveFrozenStrategyPipeline:
    """Manages frozen model loading, feature calculation, and shadow inference."""

    def __init__(self, model_seed: int = 42) -> None:
        self.model_seed = model_seed
        self._model: Optional[RandomForestBaseline] = None
        self._canonical_feature_names: List[str] = []
        self._classes: List[float] = [-1.0, 0.0, 1.0]
        self._short_idx = 0
        self._neutral_idx = 1
        self._long_idx = 2
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def canonical_feature_names(self) -> List[str]:
        return list(self._canonical_feature_names)

    @property
    def confidence_threshold(self) -> float:
        return FROZEN_CONFIDENCE_THRESHOLD

    def load_canonical_frozen_model(self) -> None:
        """Deterministically fit and lock the canonical Phase 12 Random Forest baseline."""
        logger.info("Loading canonical frozen strategy model on Training partition...")
        splits = split_dataset(
            assemble_dataset(target_column="direction_4"),
            SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4),
        )
        self._canonical_feature_names = list(splits.feature_names)
        if len(self._canonical_feature_names) != CANONICAL_FEATURE_COUNT:
            raise ValueError(
                f"Feature count mismatch: expected {CANONICAL_FEATURE_COUNT}, "
                f"got {len(self._canonical_feature_names)}"
            )

        rf = RandomForestBaseline(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=self.model_seed,
            n_jobs=-1,
        )
        rf.fit(splits.train.X, splits.train.y)
        self._model = rf

        classes = list(rf.classes_)
        self._classes = classes
        self._short_idx = classes.index(-1.0)
        self._neutral_idx = classes.index(0.0)
        self._long_idx = classes.index(1.0)
        self._is_loaded = True
        logger.info(
            f"Frozen model loaded successfully: classes={classes}, "
            f"features={len(self._canonical_feature_names)}"
        )

    def extract_features(self, buffer: M15CandleBuffer) -> pd.DataFrame:
        """Compute the 80 canonical features from completed candles."""
        if len(buffer) < MIN_WARMUP_BARS:
            raise FeatureVectorValidationError(
                f"Insufficient warmup bars: {len(buffer)} < {MIN_WARMUP_BARS}"
            )

        df_raw = buffer.to_dataframe()
        feat_df, feat_defs = build_feature_pipeline(
            df_raw, drop_warmup=False, include_raw_columns=False
        )

        if len(feat_defs) != CANONICAL_FEATURE_COUNT:
            raise FeatureVectorValidationError(
                f"Feature definition count mismatch: {len(feat_defs)} != {CANONICAL_FEATURE_COUNT}"
            )

        # Align with exact canonical feature columns
        aligned_df = feat_df[self._canonical_feature_names]
        return aligned_df

    def evaluate_latest_completed_bar(
        self,
        buffer: M15CandleBuffer,
        evaluation_timestamp_utc: datetime,
    ) -> ShadowPredictionRecord:
        """Evaluate point-in-time features on the latest completed candle.

        Strict invariants:
        - Only uses completed candle data.
        - Verifies 80 features.
        - Verifies 0 NaNs and 0 Infs.
        - Calculates directional confidence max(P_long, P_short).
        - Enforces frozen threshold = 0.60.
        """
        if not self._is_loaded or self._model is None:
            raise RuntimeError("Frozen strategy model is not loaded.")

        if len(buffer) < MIN_WARMUP_BARS:
            raise FeatureVectorValidationError(
                f"Insufficient warmup: {len(buffer)} bars < {MIN_WARMUP_BARS}"
            )

        latest_bar = buffer.bars[-1]
        candle_close_time = latest_bar.timestamp_utc + timedelta(minutes=M15_TIMEFRAME_MINUTES)
        if candle_close_time > evaluation_timestamp_utc:
            raise FeatureVectorValidationError(
                f"Lookahead leakage detected: candle close {candle_close_time} > "
                f"evaluation time {evaluation_timestamp_utc}"
            )

        # 1. Feature Extraction
        try:
            aligned_features = self.extract_features(buffer)
        except FeatureVectorValidationError:
            raise
        except Exception as exc:
            raise FeatureVectorValidationError(f"Feature extraction failed: {exc}") from exc
        X_latest = aligned_features.iloc[[-1]]

        # 2. Strict Schema Validation
        if X_latest.shape[1] != CANONICAL_FEATURE_COUNT:
            raise FeatureVectorValidationError(
                f"Feature vector length {X_latest.shape[1]} != {CANONICAL_FEATURE_COUNT}"
            )

        nan_count = int(X_latest.isna().sum().sum())
        if nan_count > 0:
            raise FeatureVectorValidationError(
                f"Feature vector contains {nan_count} NaNs on bar {latest_bar.timestamp_utc}"
            )

        inf_count = int(np.isinf(X_latest.values).sum())
        if inf_count > 0:
            raise FeatureVectorValidationError(
                f"Feature vector contains {inf_count} infinite values"
            )

        # 3. Model Inference
        try:
            probs = self._model.predict_proba(X_latest)[0]
        except Exception as exc:
            raise RuntimeError(f"Model inference failed: {exc}") from exc
        p_short = float(probs[self._short_idx])
        p_neutral = float(probs[self._neutral_idx])
        p_long = float(probs[self._long_idx])

        max_idx = int(np.argmax(probs))
        pred_class = float(self._classes[max_idx])
        class_names = {-1.0: "SHORT", 0.0: "NEUTRAL", 1.0: "LONG"}
        pred_name = class_names.get(pred_class, "UNKNOWN")

        # Directional confidence definition: max(P_LONG, P_SHORT)
        directional_confidence = max(p_short, p_long)

        # 4. Signal Filtering against Frozen Threshold (0.60)
        signal_emitted = "NONE"
        reason = "CONFIDENCE_BELOW_THRESHOLD"
        shadow_sig: Optional[Dict[str, Any]] = None

        if pred_class == 1.0 and directional_confidence >= FROZEN_CONFIDENCE_THRESHOLD:
            signal_emitted = "SHADOW_BUY"
            reason = "CONFIDENCE_THRESHOLD_MET"
            shadow_sig = {
                "action": "BUY",
                "confidence": directional_confidence,
                "reference_price": latest_bar.close,
                "timestamp_utc": evaluation_timestamp_utc.isoformat(),
                "execution_status": "SHADOW_ONLY_NO_TRADE",
            }
        elif pred_class == -1.0 and directional_confidence >= FROZEN_CONFIDENCE_THRESHOLD:
            signal_emitted = "SHADOW_SELL"
            reason = "CONFIDENCE_THRESHOLD_MET"
            shadow_sig = {
                "action": "SELL",
                "confidence": directional_confidence,
                "reference_price": latest_bar.close,
                "timestamp_utc": evaluation_timestamp_utc.isoformat(),
                "execution_status": "SHADOW_ONLY_NO_TRADE",
            }
        elif pred_class == 0.0:
            signal_emitted = "NONE"
            reason = "PREDICTED_NEUTRAL_HOLD"

        return ShadowPredictionRecord(
            timestamp_utc=evaluation_timestamp_utc.isoformat(),
            symbol=latest_bar.symbol,
            timeframe="M15",
            candle_close_time_utc=candle_close_time.isoformat(),
            feature_count=CANONICAL_FEATURE_COUNT,
            predicted_class=pred_class,
            predicted_class_name=pred_name,
            probability_short=round(p_short, 4),
            probability_neutral=round(p_neutral, 4),
            probability_long=round(p_long, 4),
            directional_confidence=round(directional_confidence, 4),
            threshold=FROZEN_CONFIDENCE_THRESHOLD,
            signal_emitted=signal_emitted,
            shadow_signal=shadow_sig,
            reason=reason,
        )


class ShadowValidationRunner:
    """Orchestrates live read-only forward shadow validation on MetaQuotes-Demo."""

    def __init__(
        self,
        client: MT5ReadOnlyClient,
        pipeline: Optional[LiveFrozenStrategyPipeline] = None,
        warmup_bars: int = RECOMMENDED_WARMUP_BARS,
        max_staleness_seconds: float = 120.0,
    ) -> None:
        self.client = client
        self.pipeline = pipeline or LiveFrozenStrategyPipeline()
        self.warmup_bars = warmup_bars
        self.max_staleness_seconds = max_staleness_seconds
        self.buffer = M15CandleBuffer()
        self.metrics = ShadowDiagnosticsMetrics()
        self.prediction_logs: List[ShadowPredictionRecord] = []
        self._confidences: List[float] = []
        self._is_running = False

    @property
    def execution_enabled(self) -> bool:
        """Always False. Execution is permanently disabled in shadow runner."""
        return False

    def stop(self) -> None:
        """Signal the observation loop to stop cleanly."""
        self._is_running = False

    def preflight_safety_check(self) -> Dict[str, Any]:
        """Perform strict preflight safety and demo-only verification."""
        if not DEMO_ONLY:
            raise BrokerExecutionDisabledError("FATAL: DEMO_ONLY invariant breached.")

        if not self.client.is_connected:
            raise MT5ConnectionError("MT5 terminal is not connected.")

        acc = self.client.get_account_metadata()
        term = self.client.get_terminal_metadata()
        sym = self.client.get_symbol_specification("EURUSD")
        tick = self.client.get_latest_tick("EURUSD")

        # Live account fail-closed gate
        if acc.trade_mode == 2:
            raise LiveAccountForbiddenError("FATAL: Live account mode detected (trade_mode == 2).")
        if acc.trade_mode != ACCOUNT_TRADE_MODE_DEMO or not acc.is_demo:
            raise LiveAccountForbiddenError(f"Non-demo account mode: {acc.trade_mode}")

        # Market open check
        if getattr(sym, "trade_mode", 1) == 0:
            raise MarketClosedError(f"Market closed: trade_mode=0 for symbol {sym.symbol}")

        now_utc = datetime.now(timezone.utc)
        staleness = (now_utc - tick.timestamp_utc).total_seconds()
        if staleness > self.max_staleness_seconds:
            raise StaleDataError(
                f"Tick data is stale: {staleness:.2f}s > {self.max_staleness_seconds}s limit"
            )

        positions = self.client.get_open_positions()
        if len(positions) > 0:
            raise PreflightSafetyError(
                f"Expected 0 open positions during preflight, found {len(positions)}"
            )

        data_fresh = staleness <= self.max_staleness_seconds

        return {
            "broker": acc.company,
            "server": acc.server,
            "account_mode": f"DEMO ({acc.trade_mode})",
            "is_demo": acc.is_demo,
            "terminal_connected": term.connected,
            "trade_allowed": term.trade_allowed,
            "symbol": sym.symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.spread,
            "tick_timestamp_utc": tick.timestamp_utc.isoformat(),
            "data_age_seconds": round(staleness, 2),
            "data_fresh": data_fresh,
            "open_positions_count": len(positions),
            "execution_enabled": self.execution_enabled,
        }

    def warmup(self) -> int:
        """Fetch historical M15 bars from MT5 and initialize completed candle buffer."""
        logger.info(f"Fetching {self.warmup_bars} historical M15 bars for warmup...")
        raw_bars = self.client.get_historical_bars(
            symbol="EURUSD",
            timeframe=MT5Timeframe.M15,
            count=self.warmup_bars,
        )
        now_utc = datetime.now(timezone.utc)
        self.buffer.populate_from_history(raw_bars, current_time_utc=now_utc)
        self.metrics.m15_bars_loaded = len(self.buffer)

        if len(self.buffer) < MIN_WARMUP_BARS:
            raise ValueError(
                f"Warmup failed: loaded {len(self.buffer)} < {MIN_WARMUP_BARS} required bars."
            )

        logger.info(
            f"Warmup successful: {len(self.buffer)} completed M15 bars buffered. "
            f"Latest completed bar: {self.buffer.latest_timestamp}"
        )
        return len(self.buffer)

    def run_single_evaluation(self, evaluation_time: datetime) -> ShadowPredictionRecord:
        """Run point-in-time feature extraction and frozen model inference."""
        record = self.pipeline.evaluate_latest_completed_bar(
            self.buffer,
            evaluation_timestamp_utc=evaluation_time,
        )
        self.prediction_logs.append(record)
        self._confidences.append(record.directional_confidence)

        # Update diagnostics counters
        self.metrics.model_evaluations += 1
        self.metrics.predictions += 1
        if record.predicted_class == 1.0:
            self.metrics.long_predictions += 1
        elif record.predicted_class == -1.0:
            self.metrics.short_predictions += 1
        else:
            self.metrics.neutral_predictions += 1

        if record.directional_confidence >= FROZEN_CONFIDENCE_THRESHOLD:
            self.metrics.count_confidence_ge_0_60 += 1
            self.metrics.signals_emitted += 1
            if record.signal_emitted == "SHADOW_BUY":
                self.metrics.long_signals += 1
            elif record.signal_emitted == "SHADOW_SELL":
                self.metrics.short_signals += 1
        else:
            self.metrics.count_confidence_lt_0_60 += 1
            self.metrics.signals_rejected_by_threshold += 1

        self.metrics.min_confidence = round(float(np.min(self._confidences)), 4)
        self.metrics.max_confidence = round(float(np.max(self._confidences)), 4)
        self.metrics.mean_confidence = round(float(np.mean(self._confidences)), 4)
        self.metrics.median_confidence = round(float(np.median(self._confidences)), 4)

        return record

    def check_for_new_completed_candles(self, now_utc: datetime) -> int:
        """Query MT5 for recent M15 bars and append newly completed candles."""
        if self.buffer.latest_timestamp is not None:
            next_close_time = self.buffer.latest_timestamp + timedelta(
                minutes=2 * M15_TIMEFRAME_MINUTES
            )
            if now_utc < next_close_time:
                # Next candle cannot possibly have completed yet
                return 0

        # Query last 3 bars
        recent_bars = self.client.get_historical_bars(
            symbol="EURUSD",
            timeframe=MT5Timeframe.M15,
            count=3,
        )
        newly_completed = 0
        for bar in recent_bars:
            if self.buffer.add_candle(bar, current_time_utc=now_utc):
                newly_completed += 1
                self.metrics.m15_bars_completed_during_run += 1
                logger.info(
                    f"New completed M15 candle appended: {bar.timestamp_utc} "
                    f"(Open: {bar.open}, Close: {bar.close})"
                )
        return newly_completed

    def run_shadow_observation(
        self,
        duration_seconds: float = 7200.0,
        poll_interval_seconds: float = 1.0,
        max_evaluations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute shadow forward observation session without trade execution."""
        self._is_running = True
        session_start_time = datetime.now(timezone.utc)
        t0 = time.time()

        # 1. Preflight Inspection
        preflight = self.preflight_safety_check()
        logger.info(f"Preflight passed: Broker {preflight['broker']}, Server {preflight['server']}")

        # 2. Model Loading
        if not self.pipeline.is_loaded:
            self.pipeline.load_canonical_frozen_model()

        # 3. Warm-up
        self.warmup()

        # 4. Initial Static Smoke Test Evaluation on latest completed bar
        logger.info("Executing initial static smoke test evaluation...")
        initial_eval = self.run_single_evaluation(evaluation_time=session_start_time)
        print(f"\n[{session_start_time.strftime('%H:%M:%S')} UTC] INITIAL SMOKE EVALUATION:")
        print(f"  Candle Close: {initial_eval.candle_close_time_utc}")
        print(
            f"  Prediction:   {initial_eval.predicted_class_name} ({initial_eval.predicted_class})"
        )
        print(
            f"  Probabilities: SHORT={initial_eval.probability_short:.4f}, "
            f"NEUTRAL={initial_eval.probability_neutral:.4f}, "
            f"LONG={initial_eval.probability_long:.4f}"
        )
        print(f"  Directional Confidence: {initial_eval.directional_confidence:.4f}")
        print(f"  Threshold:              {initial_eval.threshold:.2f}")
        print(f"  Signal:                 {initial_eval.signal_emitted}")
        print("  Execution:              DISABLED (SHADOW ONLY)\n")

        # 5. Observation Loop
        last_heartbeat = 0.0
        try:
            while self._is_running:
                now_utc = datetime.now(timezone.utc)
                elapsed = time.time() - t0

                if duration_seconds is not None and elapsed >= duration_seconds:
                    break
                if (
                    max_evaluations is not None
                    and self.metrics.model_evaluations >= max_evaluations
                ):
                    break

                # Market Data Freshness Check
                try:
                    tick = self.client.get_latest_tick("EURUSD")
                    staleness = (now_utc - tick.timestamp_utc).total_seconds()
                    if staleness > self.max_staleness_seconds:
                        self.metrics.stale_data_events += 1
                        time.sleep(poll_interval_seconds)
                        continue
                except Exception as exc:
                    logger.warning(f"Tick query error: {exc}")
                    time.sleep(poll_interval_seconds)
                    continue

                if elapsed - last_heartbeat >= 300.0:
                    print(
                        f"[{now_utc.strftime('%H:%M:%S')} UTC] HEARTBEAT: "
                        f"Elapsed {elapsed:.0f}s / {duration_seconds:.0f}s | "
                        f"Bid: {tick.bid:.5f} | Ask: {tick.ask:.5f} | "
                        f"Age: {staleness:.2f}s | Evals: {self.metrics.model_evaluations}",
                        flush=True,
                    )
                    last_heartbeat = elapsed

                # Check for candle completion
                try:
                    new_candles = self.check_for_new_completed_candles(now_utc)
                    if new_candles > 0:
                        eval_rec = self.run_single_evaluation(evaluation_time=now_utc)
                        print(
                            f"[{now_utc.strftime('%H:%M:%S')} UTC] NEW COMPLETED M15 CANDLE:",
                            flush=True,
                        )
                        print(f"  Candle Close: {eval_rec.candle_close_time_utc}", flush=True)
                        print(
                            f"  Prediction:   {eval_rec.predicted_class_name} "
                            f"({eval_rec.predicted_class})",
                            flush=True,
                        )
                        print(
                            f"  Probabilities: SHORT={eval_rec.probability_short:.4f}, "
                            f"NEUTRAL={eval_rec.probability_neutral:.4f}, "
                            f"LONG={eval_rec.probability_long:.4f}",
                            flush=True,
                        )
                        print(
                            f"  Directional Confidence: {eval_rec.directional_confidence:.4f}",
                            flush=True,
                        )
                        print(f"  Signal:                 {eval_rec.signal_emitted}", flush=True)
                        print("  Execution:              DISABLED\n", flush=True)
                except Exception as exc:
                    logger.error(f"Error during candle check / inference: {exc}")
                    self.metrics.model_evaluation_failures += 1

                time.sleep(poll_interval_seconds)
        finally:
            self._is_running = False

        session_end_time = datetime.now(timezone.utc)
        actual_duration = round((session_end_time - session_start_time).total_seconds(), 2)

        # 6. Final Reconciliation Audit (Zero Positions)
        final_positions = self.client.get_open_positions()

        # Categorize outcome
        if self.metrics.model_evaluations == 0:
            root_cause_cat = "CATEGORY A: MODEL NEVER EVALUATED"
        elif self.metrics.feature_vector_failures > 0:
            root_cause_cat = "CATEGORY B: MODEL EVALUATED BUT FEATURE PIPELINE INVALID"
        elif self.metrics.max_confidence < FROZEN_CONFIDENCE_THRESHOLD:
            root_cause_cat = "CATEGORY C: MODEL EVALUATED AND CONFIDENCE < 0.60"
        else:
            root_cause_cat = "CATEGORY D: MODEL EVALUATED AND SOME CONFIDENCE >= 0.60"

        summary = {
            "session_start_utc": session_start_time.isoformat(),
            "session_end_utc": session_end_time.isoformat(),
            "duration_seconds": actual_duration,
            "target_duration_seconds": duration_seconds,
            "market_open_status": "OPEN",
            "broker": preflight["broker"],
            "server": preflight["server"],
            "account_mode": preflight["account_mode"],
            "symbol": "EURUSD",
            "timeframe": "M15",
            "execution_enabled": False,
            "is_live": False,
            "model_identity": {
                "model_family": "RandomForestBaseline (scikit-learn RandomForestClassifier)",
                "target": "direction_4 (H=4 M15 bars / 60 minutes)",
                "parameters": {
                    "class_weight": "balanced",
                    "max_depth": 10,
                    "min_samples_leaf": 20,
                    "n_estimators": 100,
                    "random_state": self.pipeline.model_seed,
                },
                "training_bars": 69937,
                "classes": self.pipeline._classes,
            },
            "feature_identity": {
                "feature_count": CANONICAL_FEATURE_COUNT,
                "feature_names": self.pipeline.canonical_feature_names,
                "warmup_rows_required": MIN_WARMUP_BARS,
            },
            "threshold": FROZEN_CONFIDENCE_THRESHOLD,
            "metrics": {
                "m15_bars_loaded": self.metrics.m15_bars_loaded,
                "m15_bars_completed_during_run": self.metrics.m15_bars_completed_during_run,
                "model_evaluations": self.metrics.model_evaluations,
                "predictions": self.metrics.predictions,
                "long_predictions": self.metrics.long_predictions,
                "short_predictions": self.metrics.short_predictions,
                "neutral_predictions": self.metrics.neutral_predictions,
                "min_confidence": self.metrics.min_confidence,
                "max_confidence": self.metrics.max_confidence,
                "mean_confidence": self.metrics.mean_confidence,
                "median_confidence": self.metrics.median_confidence,
                "count_confidence_ge_0_60": self.metrics.count_confidence_ge_0_60,
                "count_confidence_lt_0_60": self.metrics.count_confidence_lt_0_60,
                "signals_emitted": self.metrics.signals_emitted,
                "long_signals": self.metrics.long_signals,
                "short_signals": self.metrics.short_signals,
                "signals_rejected_by_threshold": self.metrics.signals_rejected_by_threshold,
                "feature_vector_failures": self.metrics.feature_vector_failures,
                "model_evaluation_failures": self.metrics.model_evaluation_failures,
                "stale_data_events": self.metrics.stale_data_events,
                "orders_submitted": 0,
                "fills": 0,
                "open_positions": len(final_positions),
                "real_money_orders": 0,
            },
            "root_cause_classification": root_cause_cat,
            "prediction_records": [
                {
                    "timestamp_utc": p.timestamp_utc,
                    "candle_close_time_utc": p.candle_close_time_utc,
                    "predicted_class": p.predicted_class,
                    "predicted_class_name": p.predicted_class_name,
                    "probability_short": p.probability_short,
                    "probability_neutral": p.probability_neutral,
                    "probability_long": p.probability_long,
                    "directional_confidence": p.directional_confidence,
                    "signal_emitted": p.signal_emitted,
                    "reason": p.reason,
                }
                for p in self.prediction_logs
            ],
        }
        return summary


__all__ = [
    "CandleIngestionError",
    "FeatureVectorValidationError",
    "LiveFrozenStrategyPipeline",
    "M15CandleBuffer",
    "ShadowDiagnosticsMetrics",
    "ShadowPredictionRecord",
    "ShadowValidationRunner",
]
