"""
BigQuery data loader for DRL training.

Fetches campaign state features from BigQuery, normalizes them, and returns
numpy arrays ready for the DRL environment.

Configuration via environment variables:
  DRL_BQ_PROJECT_ID  (default: from GCP credentials)
  DRL_BQ_TABLE       (default: ad_metrics.campaign_states)
  GOOGLE_APPLICATION_CREDENTIALS  (standard GCP env var)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    _HAS_BQ = True
except ImportError:
    _HAS_BQ = False


DEFAULT_FEATURES: List[str] = [
    # Core metrics (6)
    "ctr",
    "cvr",
    "cpa",
    "roas",
    "cpc",
    "cpm",
    # Volume (3)
    "impressions",
    "clicks",
    "spend_velocity",
    # Trends (8)
    "ctr_trend_7d",
    "cvr_trend_7d",
    "roas_trend_7d",
    "cpa_trend_7d",
    "spend_trend_7d",
    "impressions_trend_7d",
    "clicks_trend_7d",
    "conversions_trend_7d",
    # Competitive (3)
    "impression_share",
    "auction_pressure",
    "competitive_position",
    # ML-derived (7)
    "audience_quality_score",
    "creative_fatigue_score",
    "predicted_cvr",
    "predicted_ltv",
    "predicted_revenue",
    "propensity_score",
    "engagement_score",
    # Context/temporal (6)
    "campaign_age_days",
    "budget_utilization",
    "day_of_week",
    "hour_of_day",
    "is_weekend",
    "is_holiday_season",
    # Spend features (3)  — indices 33-35
    "log_daily_spend",
    "log_total_campaign_spend",
    "log_daily_budget",
    # Audience segmentation (3)  — indices 36-38
    "segment_count",
    "top_segment_roas",
    "avg_frequency",
    # Constraint features (3)  — indices 39-41
    "target_cpa_norm",
    "min_roas_norm",
    "daily_budget_limit_norm",
]

FEATURE_ALIASES: Dict[str, str] = {
    "is_holiday": "is_holiday_season",
}


@dataclass
class BigQueryDataLoader:
    """
    Load and normalize DRL state features from BigQuery.

    Example:
        loader = BigQueryDataLoader(
            project_id="my-project",
            table="ad_metrics.campaign_states",
        )
        states, features = loader.load_states()
    """

    project_id: str = ""
    table: str = ""
    credentials_path: str = ""
    features: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))

    _mean: Optional[np.ndarray] = None
    _std: Optional[np.ndarray] = None

    def __post_init__(self):
        if not self.project_id:
            self.project_id = os.environ.get("DRL_BQ_PROJECT_ID", "")
        if not self.table:
            self.table = os.environ.get("DRL_BQ_TABLE", "ad_metrics.campaign_states")
        if not self.credentials_path:
            self.credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    def _client(self) -> "bigquery.Client":
        if not _HAS_BQ:
            raise ImportError(
                "google-cloud-bigquery is required. Install with: "
                "pip install google-cloud-bigquery"
            )
        if self.credentials_path:
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            return bigquery.Client(project=self.project_id, credentials=creds)
        # Fall back to Application Default Credentials
        return bigquery.Client(project=self.project_id or None)

    def _resolve_features(self, available: Iterable[str]) -> Tuple[List[str], List[str]]:
        """
        Resolve feature list against available columns with aliases.
        Returns (resolved_features, missing_features).
        """
        available_set = set(available)
        resolved = []
        missing = []

        for feature in self.features:
            if feature in available_set:
                resolved.append(feature)
                continue
            alias = FEATURE_ALIASES.get(feature)
            if alias and alias in available_set:
                resolved.append(alias)
            else:
                missing.append(feature)

        return resolved, missing

    def fetch_dataframe(self, limit: Optional[int] = None, strict: bool = False) -> pd.DataFrame:
        """
        Fetch the feature dataframe from BigQuery.

        Args:
            limit: Optional row limit for sampling.
            strict: If True, raise if any requested features are missing.
        """
        client = self._client()
        table_ref = f"{self.project_id}.{self.table}" if self.project_id else self.table
        table = client.get_table(table_ref)
        resolved, missing = self._resolve_features([f.name for f in table.schema])
        if missing and strict:
            raise ValueError(f"Missing features in BigQuery table: {missing}")

        columns = ", ".join(resolved)
        query = f"SELECT {columns} FROM `{table_ref}`"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        df = client.query(query).to_dataframe()
        self.features = resolved
        return df

    def fit_normalizer(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Compute and store mean/std for standardization."""
        values = df[self.features].to_numpy(dtype=np.float32)
        self._mean = values.mean(axis=0)
        self._std = values.std(axis=0)
        self._std = np.where(self._std == 0, 1.0, self._std)
        return self._mean, self._std

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Normalize dataframe to numpy array using stored mean/std."""
        df_features = df[self.features].copy()
        df_features = df_features.apply(
            lambda col: col.fillna(col.mean()) if col.isnull().any() else col
        )
        if self._mean is None or self._std is None:
            self.fit_normalizer(df_features)
        values = df_features.to_numpy(dtype=np.float32)
        return (values - self._mean) / self._std

    def load_states(
        self, limit: Optional[int] = None, strict: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fetch and normalize states as numpy array.

        Returns:
            (states, feature_list)
        """
        df = self.fetch_dataframe(limit=limit, strict=strict)
        states = self.transform(df)
        return states, list(self.features)
