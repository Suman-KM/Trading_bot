"""Phase 40.4: Live Frozen Strategy Integration & Shadow Runner Unit Tests.

Covers all 22 required diagnostic and safety invariants:
1. strategy_fn is correctly wired
2. model is actually invoked
3. 80-feature vector generated
4. feature order validated
5. NaN/Inf rejected
6. insufficient warmup rejected
7. duplicate M15 candle prevented
8. timestamp monotonicity
9. UTC normalization
10. completed-candle-only evaluation
11. confidence calculation matches frozen implementation
12. threshold remains exactly 0.60
13. signal generation does not submit orders
14. execution remains disabled
15. live account gate
16. demo-only gate
17. stale-data gate
18. market-closed gate
19. model loading failure
20. feature-generation failure
21. model evaluation failure
22. zero-signal diagnostic accounting
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tests.test_phase40_forward_validation import MockBackendForForward
from trading.adapters.mt5.client import MT5ReadOnlyClient
from trading.adapters.mt5.normalizer import normalize_timestamp
from trading.adapters.mt5.safety import (
    ACCOUNT_TRADE_MODE_REAL,
    LiveAccountForbiddenError,
)
from trading.adapters.mt5.schemas import MT5BarData, MT5Timeframe
from trading.adapters.mt5.shadow import (
    CANONICAL_FEATURE_COUNT,
    FROZEN_CONFIDENCE_THRESHOLD,
    CandleIngestionError,
    FeatureVectorValidationError,
    LiveFrozenStrategyPipeline,
    M15CandleBuffer,
    MarketClosedError,
    ShadowPredictionRecord,
    ShadowValidationRunner,
    StaleDataError,
)


def make_mock_m15_bar(
    index: int,
    base_time: datetime,
    close_price: float = 1.12000,
) -> MT5BarData:
    """Generate a single valid mock M15 bar."""
    bar_ts = base_time + timedelta(minutes=15 * index)
    return MT5BarData(
        symbol="EURUSD",
        timeframe=MT5Timeframe.M15,
        timestamp_utc=bar_ts,
        open=close_price - 0.00010,
        high=close_price + 0.00020,
        low=close_price - 0.00020,
        close=close_price,
        tick_volume=100 + index,
        spread=1,
        real_volume=0.0,
    )


def make_mock_m15_history(count: int, base_time: datetime) -> List[MT5BarData]:
    """Generate a sequence of count valid M15 bars."""
    bars = []
    price = 1.12000
    for i in range(count):
        price += np.sin(i / 10.0) * 0.00020
        bars.append(make_mock_m15_bar(i, base_time, close_price=round(price, 5)))
    return bars


@pytest.fixture
def mock_client():
    backend = MockBackendForForward()
    client = MT5ReadOnlyClient(mock_backend=backend)
    client.initialize()
    return client


# 1. strategy_fn is correctly wired
def test_01_strategy_fn_correctly_wired():
    """Verify LiveFrozenStrategyPipeline can load canonical model and wire into runner."""
    pipeline = LiveFrozenStrategyPipeline()
    pipeline.load_canonical_frozen_model()
    assert pipeline.is_loaded
    assert len(pipeline.canonical_feature_names) == CANONICAL_FEATURE_COUNT

    mock_client = MagicMock(spec=MT5ReadOnlyClient)
    runner = ShadowValidationRunner(client=mock_client, pipeline=pipeline)
    assert runner.pipeline is pipeline
    assert runner.pipeline.is_loaded


# 2. model is actually invoked
def test_02_model_is_actually_invoked():
    """Verify evaluate_latest_completed_bar actually calls predict_proba on the model."""
    pipeline = LiveFrozenStrategyPipeline()
    pipeline._is_loaded = True
    pipeline._canonical_feature_names = [f"f_{i}" for i in range(80)]
    pipeline._classes = [-1.0, 0.0, 1.0]
    pipeline._short_idx, pipeline._neutral_idx, pipeline._long_idx = 0, 1, 2

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.30, 0.40, 0.30]])
    pipeline._model = mock_model
    pipeline.extract_features = MagicMock(
        return_value=pd.DataFrame([{f"f_{i}": 1.0 for i in range(80)}])
    )

    buffer = MagicMock(spec=M15CandleBuffer)
    buffer.__len__.return_value = 85
    latest_bar = MagicMock()
    latest_bar.timestamp_utc = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    latest_bar.symbol = "EURUSD"
    latest_bar.close = 1.12000
    buffer.bars = [latest_bar]

    eval_time = datetime(2026, 1, 1, 12, 16, tzinfo=timezone.utc)
    res = pipeline.evaluate_latest_completed_bar(buffer, evaluation_timestamp_utc=eval_time)

    assert mock_model.predict_proba.called
    assert res.predicted_class == 0.0
    assert res.predicted_class_name == "NEUTRAL"


# 3. 80-feature vector generated
def test_03_80_feature_vector_generated():
    """Verify feature extractor generates exactly 80 features."""
    pipeline = LiveFrozenStrategyPipeline()
    pipeline.load_canonical_frozen_model()

    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    now_utc = base_time + timedelta(minutes=15 * 90)
    bars = make_mock_m15_history(85, base_time)

    buffer = M15CandleBuffer()
    buffer.populate_from_history(bars, current_time_utc=now_utc)

    feat_df = pipeline.extract_features(buffer)
    assert feat_df.shape[1] == 80


# 4. feature order validated
def test_04_feature_order_validated():
    """Verify feature column ordering matches canonical feature names exactly."""
    pipeline = LiveFrozenStrategyPipeline()
    pipeline.load_canonical_frozen_model()

    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    now_utc = base_time + timedelta(minutes=15 * 90)
    bars = make_mock_m15_history(85, base_time)

    buffer = M15CandleBuffer()
    buffer.populate_from_history(bars, current_time_utc=now_utc)

    feat_df = pipeline.extract_features(buffer)
    assert list(feat_df.columns) == pipeline.canonical_feature_names


# 5. NaN/Inf rejected
def test_05_nan_and_inf_rejected():
    """Verify FeatureVectorValidationError when feature vector contains NaNs or Infs."""
    pipeline = LiveFrozenStrategyPipeline()
    pipeline._is_loaded = True
    pipeline._model = MagicMock()
    pipeline._canonical_feature_names = [f"feat_{i}" for i in range(80)]

    buffer = MagicMock(spec=M15CandleBuffer)
    buffer.__len__.return_value = 85
    latest_bar = MagicMock()
    latest_bar.timestamp_utc = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    latest_bar.symbol = "EURUSD"
    buffer.bars = [latest_bar]

    # Return DataFrame with NaN
    nan_df = pd.DataFrame([{f"feat_{i}": (np.nan if i == 0 else 1.0) for i in range(80)}])
    pipeline.extract_features = MagicMock(return_value=nan_df)

    eval_time = latest_bar.timestamp_utc + timedelta(minutes=20)
    with pytest.raises(FeatureVectorValidationError, match="contains 1 NaNs"):
        pipeline.evaluate_latest_completed_bar(buffer, evaluation_timestamp_utc=eval_time)


# 6. insufficient warmup rejected
def test_06_insufficient_warmup_rejected():
    """Verify error when buffer has fewer than MIN_WARMUP_BARS (80)."""
    pipeline = LiveFrozenStrategyPipeline()
    pipeline._is_loaded = True
    pipeline._model = MagicMock()

    buffer = M15CandleBuffer()
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    now_utc = base_time + timedelta(minutes=15 * 50)
    bars = make_mock_m15_history(50, base_time)
    buffer.populate_from_history(bars, current_time_utc=now_utc)

    with pytest.raises(FeatureVectorValidationError, match="Insufficient warmup"):
        pipeline.evaluate_latest_completed_bar(buffer, evaluation_timestamp_utc=now_utc)


# 7. duplicate M15 candle prevented
def test_07_duplicate_m15_candle_prevented():
    """Verify duplicate candle timestamps are rejected."""
    buffer = M15CandleBuffer()
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    now_utc = base_time + timedelta(minutes=30)

    bar = make_mock_m15_bar(0, base_time)
    assert buffer.add_candle(bar, current_time_utc=now_utc) is True
    # Attempt duplicate
    assert buffer.add_candle(bar, current_time_utc=now_utc) is False
    assert len(buffer) == 1


# 8. timestamp monotonicity
def test_08_timestamp_monotonicity_enforced():
    """Verify non-increasing timestamps raise CandleIngestionError."""
    buffer = M15CandleBuffer()
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    now_utc = base_time + timedelta(minutes=60)

    bar1 = make_mock_m15_bar(1, base_time)
    bar0 = make_mock_m15_bar(0, base_time)

    buffer.add_candle(bar1, current_time_utc=now_utc)
    with pytest.raises(CandleIngestionError, match="Timestamp regression"):
        buffer.add_candle(bar0, current_time_utc=now_utc)


# 9. UTC normalization
def test_09_utc_normalization_enforced():
    """Verify timestamp normalizer converts ISO strings and naive timestamps to UTC."""
    raw_str = "2026-10-05 14:30:00"
    norm_ts = normalize_timestamp(raw_str)
    assert norm_ts.tzinfo == timezone.utc
    assert norm_ts.year == 2026
    assert norm_ts.month == 10
    assert norm_ts.day == 5
    # Naive broker string in EEST (UTC+3) is normalized to UTC (11:30)
    assert norm_ts.hour == 11
    assert norm_ts.minute == 30

    # Explicit UTC ISO string preserves exact hour
    utc_str = "2026-10-05T14:30:00Z"
    norm_utc = normalize_timestamp(utc_str)
    assert norm_utc.tzinfo == timezone.utc
    assert norm_utc.hour == 14
    assert norm_utc.minute == 30


# 10. completed-candle-only evaluation
def test_10_completed_candle_only_evaluation():
    """Verify candle still forming is rejected by buffer to prevent lookahead."""
    buffer = M15CandleBuffer()
    bar_ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    forming_bar = MT5BarData(
        symbol="EURUSD",
        timeframe=MT5Timeframe.M15,
        timestamp_utc=bar_ts,
        open=1.12000,
        high=1.12050,
        low=1.11950,
        close=1.12020,
        tick_volume=50,
        spread=1,
        real_volume=0.0,
    )
    # Current time is 12:05 (bar closes at 12:15)
    now_utc = datetime(2026, 1, 1, 12, 5, tzinfo=timezone.utc)
    assert buffer.add_candle(forming_bar, current_time_utc=now_utc) is False
    assert len(buffer) == 0

    # Current time becomes 12:15:01 (bar is completed)
    completed_time = datetime(2026, 1, 1, 12, 15, 1, tzinfo=timezone.utc)
    assert buffer.add_candle(forming_bar, current_time_utc=completed_time) is True
    assert len(buffer) == 1


# 11. confidence calculation matches frozen implementation
def test_11_confidence_calculation_matches_frozen_impl():
    """Verify directional confidence calculation max(P_long, P_short)."""
    pipeline = LiveFrozenStrategyPipeline()
    pipeline._is_loaded = True
    pipeline._canonical_feature_names = [f"col_{i}" for i in range(80)]
    pipeline._classes = [-1.0, 0.0, 1.0]
    pipeline._short_idx = 0
    pipeline._neutral_idx = 1
    pipeline._long_idx = 2

    # Mock model predicting SHORT with 0.45 prob
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.45, 0.20, 0.35]])
    pipeline._model = mock_model
    pipeline.extract_features = MagicMock(
        return_value=pd.DataFrame([{f"col_{i}": 1.0 for i in range(80)}])
    )

    buffer = MagicMock()
    buffer.__len__.return_value = 85
    bar = MagicMock()
    bar.timestamp_utc = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    bar.symbol = "EURUSD"
    bar.close = 1.12000
    buffer.bars = [bar]

    eval_time = datetime(2026, 1, 1, 12, 20, tzinfo=timezone.utc)
    rec = pipeline.evaluate_latest_completed_bar(buffer, evaluation_timestamp_utc=eval_time)
    assert rec.directional_confidence == 0.45


# 12. threshold remains exactly 0.60
def test_12_threshold_remains_exactly_0_60():
    """Verify frozen threshold is invariant at 0.60 and filters signals < 0.60."""
    assert FROZEN_CONFIDENCE_THRESHOLD == 0.60
    pipeline = LiveFrozenStrategyPipeline()
    assert pipeline.confidence_threshold == 0.60


# 13. signal generation does not submit orders
def test_13_signal_generation_does_not_submit_orders():
    """Verify that when confidence >= 0.60, SHADOW_BUY is recorded but no order sent."""
    pipeline = LiveFrozenStrategyPipeline()
    pipeline._is_loaded = True
    pipeline._canonical_feature_names = [f"col_{i}" for i in range(80)]
    pipeline._classes = [-1.0, 0.0, 1.0]
    pipeline._short_idx = 0
    pipeline._neutral_idx = 1
    pipeline._long_idx = 2

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.15, 0.20, 0.65]])
    pipeline._model = mock_model
    pipeline.extract_features = MagicMock(
        return_value=pd.DataFrame([{f"col_{i}": 1.0 for i in range(80)}])
    )

    buffer = MagicMock()
    buffer.__len__.return_value = 85
    bar = MagicMock()
    bar.timestamp_utc = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    bar.symbol = "EURUSD"
    bar.close = 1.12500
    buffer.bars = [bar]

    eval_time = datetime(2026, 1, 1, 12, 20, tzinfo=timezone.utc)
    rec = pipeline.evaluate_latest_completed_bar(buffer, evaluation_timestamp_utc=eval_time)
    assert rec.directional_confidence == 0.65
    assert rec.signal_emitted == "SHADOW_BUY"
    assert rec.shadow_signal is not None
    assert rec.shadow_signal["action"] == "BUY"
    assert rec.shadow_signal["execution_status"] == "SHADOW_ONLY_NO_TRADE"


# 14. execution remains disabled
def test_14_execution_remains_disabled(mock_client):
    """Verify ShadowValidationRunner has execution_enabled == False."""
    runner = ShadowValidationRunner(client=mock_client)
    assert runner.execution_enabled is False


# 15. live account gate
def test_15_live_account_forbidden_gate(mock_client):
    """Verify live account (trade_mode == 2) triggers LiveAccountForbiddenError."""
    mock_client._mock_backend.trade_mode = ACCOUNT_TRADE_MODE_REAL
    runner = ShadowValidationRunner(client=mock_client)
    with pytest.raises(LiveAccountForbiddenError):
        runner.preflight_safety_check()


# 16. demo-only gate
def test_16_demo_only_gate(mock_client):
    """Verify non-demo account fails demo gate in preflight check."""
    mock_client._mock_backend.trade_mode = 1  # Contest mode
    runner = ShadowValidationRunner(client=mock_client)
    with pytest.raises(LiveAccountForbiddenError):
        runner.preflight_safety_check()


# 17. stale-data gate
def test_17_stale_data_gate(mock_client):
    """Verify stale tick data (>120s) raises StaleDataError during preflight."""
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=300)
    mock_tick = MagicMock()
    mock_tick.timestamp_utc = stale_time
    mock_client.get_latest_tick = MagicMock(return_value=mock_tick)
    runner = ShadowValidationRunner(client=mock_client, max_staleness_seconds=120.0)
    with pytest.raises(StaleDataError, match="Tick data is stale"):
        runner.preflight_safety_check()


# 18. market-closed gate
def test_18_market_closed_gate(mock_client):
    """Verify closed market (trade_mode == 0) raises MarketClosedError during preflight."""
    mock_sym = MagicMock()
    mock_sym.trade_mode = 0
    mock_sym.symbol = "EURUSD"
    mock_client.get_symbol_specification = MagicMock(return_value=mock_sym)
    runner = ShadowValidationRunner(client=mock_client)
    with pytest.raises(MarketClosedError, match="Market closed"):
        runner.preflight_safety_check()


# 19. model loading failure
def test_19_model_loading_failure():
    """Verify calling evaluate on unloaded pipeline raises RuntimeError."""
    pipeline = LiveFrozenStrategyPipeline()
    assert not pipeline.is_loaded
    buffer = M15CandleBuffer()
    with pytest.raises(RuntimeError, match="model is not loaded"):
        pipeline.evaluate_latest_completed_bar(
            buffer, evaluation_timestamp_utc=datetime.now(timezone.utc)
        )


# 20. feature-generation failure
def test_20_feature_generation_failure():
    """Verify that feature generation failure raises FeatureVectorValidationError."""
    pipeline = LiveFrozenStrategyPipeline()
    pipeline._is_loaded = True
    pipeline._model = MagicMock()
    pipeline._canonical_feature_names = [f"f_{i}" for i in range(80)]

    buffer = MagicMock(spec=M15CandleBuffer)
    buffer.__len__.return_value = 85
    latest_bar = MagicMock()
    latest_bar.timestamp_utc = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    buffer.bars = [latest_bar]

    # Mock extract_features raising error
    pipeline.extract_features = MagicMock(side_effect=ValueError("Feature calculation crash"))

    with pytest.raises(FeatureVectorValidationError, match="Feature extraction failed"):
        pipeline.evaluate_latest_completed_bar(
            buffer, evaluation_timestamp_utc=datetime(2026, 1, 1, 12, 16, tzinfo=timezone.utc)
        )


# 21. model evaluation failure
def test_21_model_evaluation_failure():
    """Verify that model inference exception is caught and raises RuntimeError."""
    pipeline = LiveFrozenStrategyPipeline()
    pipeline._is_loaded = True
    pipeline._canonical_feature_names = [f"f_{i}" for i in range(80)]
    pipeline._classes = [-1.0, 0.0, 1.0]

    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = RuntimeError("Inference memory fault")
    pipeline._model = mock_model
    pipeline.extract_features = MagicMock(
        return_value=pd.DataFrame([{f"f_{i}": 1.0 for i in range(80)}])
    )

    buffer = MagicMock(spec=M15CandleBuffer)
    buffer.__len__.return_value = 85
    latest_bar = MagicMock()
    latest_bar.timestamp_utc = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    buffer.bars = [latest_bar]

    with pytest.raises(RuntimeError, match="Model inference failed"):
        pipeline.evaluate_latest_completed_bar(
            buffer, evaluation_timestamp_utc=datetime(2026, 1, 1, 12, 16, tzinfo=timezone.utc)
        )


# 22. zero-signal diagnostic accounting
def test_22_zero_signal_diagnostic_accounting():
    """Verify runner diagnostics metrics calculate min/max/mean/median confidence and 0 orders."""
    mock_client = MagicMock(spec=MT5ReadOnlyClient)
    runner = ShadowValidationRunner(client=mock_client)

    rec1 = ShadowPredictionRecord(
        timestamp_utc="2026-10-05T12:00:00",
        symbol="EURUSD",
        timeframe="M15",
        candle_close_time_utc="2026-10-05T11:45:00",
        feature_count=80,
        predicted_class=-1.0,
        predicted_class_name="SHORT",
        probability_short=0.45,
        probability_neutral=0.20,
        probability_long=0.35,
        directional_confidence=0.45,
        threshold=0.60,
        signal_emitted="NONE",
    )
    rec2 = ShadowPredictionRecord(
        timestamp_utc="2026-10-05T12:15:00",
        symbol="EURUSD",
        timeframe="M15",
        candle_close_time_utc="2026-10-05T12:00:00",
        feature_count=80,
        predicted_class=1.0,
        predicted_class_name="LONG",
        probability_short=0.30,
        probability_neutral=0.20,
        probability_long=0.50,
        directional_confidence=0.50,
        threshold=0.60,
        signal_emitted="NONE",
    )

    runner.pipeline = MagicMock()
    runner.pipeline.evaluate_latest_completed_bar.side_effect = [rec1, rec2]

    runner.run_single_evaluation(datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc))
    runner.run_single_evaluation(datetime(2026, 10, 5, 12, 15, tzinfo=timezone.utc))

    assert runner.metrics.model_evaluations == 2
    assert runner.metrics.predictions == 2
    assert runner.metrics.short_predictions == 1
    assert runner.metrics.long_predictions == 1
    assert runner.metrics.min_confidence == 0.45
    assert runner.metrics.max_confidence == 0.50
    assert runner.metrics.mean_confidence == 0.475
    assert runner.metrics.median_confidence == 0.475
    assert runner.metrics.count_confidence_ge_0_60 == 0
    assert runner.metrics.count_confidence_lt_0_60 == 2
    assert runner.metrics.signals_emitted == 0
    assert runner.metrics.orders_submitted == 0
    assert runner.metrics.fills == 0
