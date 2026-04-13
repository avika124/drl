"""
Benchmark prediction model for ad campaign optimization.

Provides campaign outcome forecasting with confidence intervals using
ridge regression and bootstrap resampling.

Inputs:
  - budget, duration, industry (optional), target metric
  - Or: a feature vector from existing campaign state

Outputs:
  - MetricForecast with (mean, p10, p90) for each outcome metric
  - Threshold recommendations based on percentile analysis

This module is designed to be called from the execution pipeline to
produce the "results forecasts" required by the P & X Model design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# Outcome metrics the model can forecast
OUTCOME_COLUMNS = [
    "impressions", "clicks", "cpc", "ctr", "cvr", "roas", "cpa",
]


@dataclass
class MetricForecast:
    """Forecast for a single metric with confidence interval."""
    metric: str
    mean: float
    p10: float  # 10th percentile (pessimistic)
    p90: float  # 90th percentile (optimistic)

    def to_dict(self) -> Dict[str, float]:
        return {
            "metric": self.metric,
            "mean": round(self.mean, 4),
            "p10": round(self.p10, 4),
            "p90": round(self.p90, 4),
        }


@dataclass
class CampaignForecast:
    """Complete forecast for a campaign with all metrics and thresholds."""
    metric_forecasts: List[MetricForecast] = field(default_factory=list)
    estimated_conversions: Optional[MetricForecast] = None
    thresholds: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "metrics": {f.metric: f.to_dict() for f in self.metric_forecasts},
            "estimated_conversions": self.estimated_conversions.to_dict() if self.estimated_conversions else None,
            "thresholds": self.thresholds,
        }


@dataclass
class LinearModel:
    """Simple ridge regression model."""
    coef: np.ndarray
    intercept: float
    resid_std: float

    def predict(self, x: np.ndarray) -> float:
        return float(x @ self.coef + self.intercept)


def fit_ridge(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> LinearModel:
    """Fit a ridge regression model (closed-form)."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    I = np.eye(Xb.shape[1])
    I[-1, -1] = 0.0  # don't regularize bias
    w = np.linalg.solve(Xb.T @ Xb + l2 * I, Xb.T @ y)
    coef, intercept = w[:-1], w[-1]
    preds = Xb @ w
    resid_std = float(np.std(y - preds))
    return LinearModel(coef=coef, intercept=float(intercept), resid_std=resid_std)


def build_models(
    X: np.ndarray,
    outcomes: Dict[str, np.ndarray],
) -> Dict[str, LinearModel]:
    """Build one ridge model per outcome metric."""
    models = {}
    for col, y in outcomes.items():
        models[col] = fit_ridge(X, y)
    return models


def bootstrap_interval(
    X: np.ndarray,
    y: np.ndarray,
    x_query: np.ndarray,
    n_boot: int = 200,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a single prediction.
    Returns (mean, p10, p90).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    preds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        model = fit_ridge(X[idx], y[idx])
        preds.append(model.predict(x_query))
    preds_arr = np.array(preds)
    return (
        float(np.mean(preds_arr)),
        float(np.percentile(preds_arr, 10)),
        float(np.percentile(preds_arr, 90)),
    )


def compute_thresholds(outcome_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute decision thresholds from historical outcome distributions."""
    thresholds = {}
    if "cpc" in outcome_data:
        thresholds["cpc_stop"] = float(np.percentile(outcome_data["cpc"], 90))
    if "ctr" in outcome_data:
        ctr = outcome_data["ctr"]
        thresholds["ctr_stop"] = float(np.percentile(ctr, 10))
        thresholds["ctr_pause_drop"] = float(np.median(ctr) * 0.8)
    if "roas" in outcome_data:
        thresholds["roas_increase_budget"] = float(np.percentile(outcome_data["roas"], 90))
    if "cpa" in outcome_data:
        thresholds["cpa_decrease_bid"] = float(np.percentile(outcome_data["cpa"], 90))
    return thresholds


class CampaignForecaster:
    """
    Forecasts campaign outcomes using ridge regression with bootstrap CIs.

    Usage:
        forecaster = CampaignForecaster()
        forecaster.fit(X_train, outcomes_train)
        forecast = forecaster.predict(x_query)
    """

    def __init__(self, n_bootstrap: int = 200):
        self.n_bootstrap = n_bootstrap
        self._models: Dict[str, LinearModel] = {}
        self._X: Optional[np.ndarray] = None
        self._outcomes: Dict[str, np.ndarray] = {}
        self._thresholds: Dict[str, float] = {}

    def fit(
        self,
        X: np.ndarray,
        outcomes: Dict[str, np.ndarray],
    ) -> None:
        """
        Fit models on training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            outcomes: Dict mapping metric name -> array of values
        """
        self._X = X
        self._outcomes = outcomes
        self._models = build_models(X, outcomes)
        self._thresholds = compute_thresholds(outcomes)

    def predict(self, x_query: np.ndarray) -> CampaignForecast:
        """
        Forecast all outcome metrics for a given feature vector.

        Args:
            x_query: Single feature vector (n_features,)

        Returns:
            CampaignForecast with per-metric (mean, p10, p90) and thresholds.
        """
        if self._X is None:
            raise RuntimeError("Call fit() before predict()")

        forecasts = []
        for metric in OUTCOME_COLUMNS:
            if metric not in self._outcomes:
                continue
            mean, p10, p90 = bootstrap_interval(
                self._X, self._outcomes[metric], x_query,
                n_boot=self.n_bootstrap,
            )
            forecasts.append(MetricForecast(metric=metric, mean=mean, p10=p10, p90=p90))

        # Derived: estimated conversions = clicks * cvr
        clicks_f = next((f for f in forecasts if f.metric == "clicks"), None)
        cvr_f = next((f for f in forecasts if f.metric == "cvr"), None)
        est_conv = None
        if clicks_f and cvr_f:
            est_conv = MetricForecast(
                metric="conversions",
                mean=clicks_f.mean * cvr_f.mean,
                p10=max(clicks_f.p10 * cvr_f.p10, 0),
                p90=max(clicks_f.p90 * cvr_f.p90, 0),
            )

        return CampaignForecast(
            metric_forecasts=forecasts,
            estimated_conversions=est_conv,
            thresholds=self._thresholds,
        )
