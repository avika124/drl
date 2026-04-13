"""
Forecast-vs-Actual Feedback Loop

Closes the loop between CampaignForecaster predictions and observed outcomes.
When actual results arrive (via OutcomeTracker), this module:
  1. Compares predicted metrics to actuals
  2. Tracks forecast accuracy over time
  3. Accumulates new (X, y) pairs and re-fits the forecaster periodically

Designed to be used alongside ContinuousLearningEngine — the same
outcome signal that drives DRL learning also refines the forecaster.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .benchmark_model import CampaignForecaster, CampaignForecast, OUTCOME_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class ForecastRecord:
    """A single forecast + its eventual actual outcome."""
    campaign_id: str
    tracking_id: str
    timestamp: str

    # What the model predicted
    forecast: CampaignForecast

    # State feature vector used to produce the forecast
    feature_vector: np.ndarray

    # Actuals (filled in when the outcome arrives)
    actuals: Optional[Dict[str, float]] = None
    outcome_timestamp: Optional[str] = None


@dataclass
class AccuracyMetrics:
    """Aggregate forecast accuracy for a single metric."""
    metric: str
    mae: float           # mean absolute error
    mape: float          # mean absolute percentage error (0-1)
    coverage_p10_p90: float  # fraction of actuals falling inside [p10, p90]
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "mae": round(self.mae, 4),
            "mape": round(self.mape, 4),
            "coverage_p10_p90": round(self.coverage_p10_p90, 4),
            "n_samples": self.n_samples,
        }


class ForecastFeedbackLoop:
    """
    Collects forecast–actual pairs, measures accuracy, and re-fits
    the CampaignForecaster when enough new data has been gathered.

    Usage:
        loop = ForecastFeedbackLoop(forecaster)

        # At optimization time:
        loop.register_forecast(campaign_id, tracking_id, forecast, features)

        # When outcome arrives:
        loop.record_actual(tracking_id, actual_metrics)

        # Periodically:
        loop.maybe_refit()
        accuracy = loop.get_accuracy()
    """

    def __init__(
        self,
        forecaster: CampaignForecaster,
        refit_every_n: int = 200,
        max_history: int = 10_000,
    ):
        """
        Args:
            forecaster: The forecaster to close the loop on.
            refit_every_n: Re-fit after this many new actuals.
            max_history: Maximum completed records to keep.
        """
        self.forecaster = forecaster
        self.refit_every_n = refit_every_n
        self.max_history = max_history

        # Pending forecasts awaiting actuals (keyed by tracking_id)
        self._pending: Dict[str, ForecastRecord] = {}

        # Completed records with both forecast and actual
        self._completed: deque[ForecastRecord] = deque(maxlen=max_history)

        # Counter of new actuals since last refit
        self._new_since_refit: int = 0

    # ── register / record ──────────────────────────────────────────

    def register_forecast(
        self,
        campaign_id: str,
        tracking_id: str,
        forecast: CampaignForecast,
        feature_vector: np.ndarray,
    ) -> None:
        """Store a forecast that will be compared against future actuals."""
        self._pending[tracking_id] = ForecastRecord(
            campaign_id=campaign_id,
            tracking_id=tracking_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            forecast=forecast,
            feature_vector=feature_vector,
        )

    def record_actual(
        self,
        tracking_id: str,
        actual_metrics: Dict[str, float],
    ) -> Optional[ForecastRecord]:
        """
        Attach actual outcomes to a pending forecast.

        Returns the completed ForecastRecord, or None if the tracking_id
        is unknown (e.g. forecast was made before this loop was active).
        """
        record = self._pending.pop(tracking_id, None)
        if record is None:
            return None

        record.actuals = actual_metrics
        record.outcome_timestamp = datetime.now(timezone.utc).isoformat()
        self._completed.append(record)
        self._new_since_refit += 1
        return record

    # ── accuracy ───────────────────────────────────────────────────

    def get_accuracy(self, last_n: Optional[int] = None) -> List[AccuracyMetrics]:
        """
        Compute per-metric accuracy over the most recent completed records.

        Args:
            last_n: Number of recent records to use (default: all).
        """
        records = list(self._completed)
        if last_n is not None:
            records = records[-last_n:]

        if not records:
            return []

        results: List[AccuracyMetrics] = []
        for metric in OUTCOME_COLUMNS:
            errors = []
            pct_errors = []
            in_band = 0
            total = 0

            for rec in records:
                if rec.actuals is None or metric not in rec.actuals:
                    continue
                actual = rec.actuals[metric]

                # Find the matching MetricForecast
                mf = next(
                    (f for f in rec.forecast.metric_forecasts if f.metric == metric),
                    None,
                )
                if mf is None:
                    continue

                total += 1
                err = abs(mf.mean - actual)
                errors.append(err)
                if actual != 0:
                    pct_errors.append(err / abs(actual))
                # Coverage: actual inside [p10, p90]?
                if mf.p10 <= actual <= mf.p90:
                    in_band += 1

            if total == 0:
                continue

            results.append(AccuracyMetrics(
                metric=metric,
                mae=float(np.mean(errors)),
                mape=float(np.mean(pct_errors)) if pct_errors else 0.0,
                coverage_p10_p90=in_band / total,
                n_samples=total,
            ))

        return results

    def get_accuracy_dict(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Return accuracy as a serialisable dict."""
        return {m.metric: m.to_dict() for m in self.get_accuracy(last_n)}

    # ── refit ──────────────────────────────────────────────────────

    def maybe_refit(self) -> bool:
        """
        Re-fit the forecaster if enough new data has accumulated.

        Returns True if a refit was performed.
        """
        if self._new_since_refit < self.refit_every_n:
            return False

        completed = [r for r in self._completed if r.actuals is not None]
        if len(completed) < 50:
            return False

        X = np.array([r.feature_vector for r in completed])
        outcomes: Dict[str, np.ndarray] = {}
        for metric in OUTCOME_COLUMNS:
            values = [r.actuals.get(metric) for r in completed]
            if all(v is not None for v in values):
                outcomes[metric] = np.array(values, dtype=float)

        if not outcomes:
            return False

        self.forecaster.fit(X, outcomes)
        self._new_since_refit = 0
        logger.info(
            f"Forecaster re-fit on {len(completed)} records "
            f"({len(outcomes)} metrics)"
        )
        return True

    # ── diagnostics ────────────────────────────────────────────────

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic info for monitoring dashboards."""
        return {
            "pending_forecasts": len(self._pending),
            "completed_records": len(self._completed),
            "new_since_refit": self._new_since_refit,
            "refit_threshold": self.refit_every_n,
            "accuracy": self.get_accuracy_dict(last_n=200),
        }
