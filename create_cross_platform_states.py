"""
Create cross-platform state vectors from raw campaign data.

===== STEP 1: P-MODEL TRAINING =====
[create_cross_platform_states]
Description: Builds 42-dimensional state vectors from raw campaign metrics (single or
  multi-platform). Output can be written to CSV or BigQuery table for cross-platform
  optimization and offline training.

Input: BigQuery table(s) or CSV with raw metrics (ctr, cvr, cpc, roas, etc.)
Output: CSV or BigQuery table with normalized state vectors -> run_cross_platform_optimizer,
  train_bigquery_offline
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# QA/Testing: Set True to enable input/output logging for traceability
_QA_IO_LOGGING = True

from .bigquery_loader import BigQueryDataLoader, DEFAULT_FEATURES
from .state_action import CampaignState

PLATFORM_ONE_HOT = ["is_google", "is_meta", "is_amazon", "is_tiktok", "is_linkedin"]


def _build_state_row(
    row: pd.Series,
    platform: str,
    total_campaign_spend: float = 0.0,
) -> np.ndarray:
    """
    Build a 42-dim state vector from a single campaign metrics row.
    """
    state = CampaignState.from_campaign_metrics(
        campaign_id=str(row.get("campaign_id", "")),
        metrics={
            "ctr": float(row.get("ctr", 0.0) or 0.0),
            "cvr": float(row.get("cvr", 0.0) or 0.0),
            "cpa": float(row.get("cpa", 0.0) or 0.0),
            "roas": float(row.get("roas", 0.0) or 0.0),
            "cpc": float(row.get("cpc", 0.0) or 0.0),
            "cpm": float(row.get("cpm", 0.0) or 0.0),
            "spend_velocity": float(row.get("spend_velocity", 0.0) or 0.0),
            "impressions": float(row.get("impressions", 0.0) or 0.0),
            "clicks": float(row.get("clicks", 0.0) or 0.0),
            "budget_utilization": float(row.get("budget_utilization", 0.0) or 0.0),
            "daily_budget": float(row.get("daily_budget", 1000) or 1000),
            "spend": float(row.get("spend", 0.0) or 0.0),
            "total_campaign_spend": total_campaign_spend,
            "ctr_trend_7d": float(row.get("ctr_trend_7d", 0.0) or 0.0),
            "cvr_trend_7d": float(row.get("cvr_trend_7d", 0.0) or 0.0),
            "roas_trend_7d": float(row.get("roas_trend_7d", 0.0) or 0.0),
            "cpa_trend_7d": float(row.get("cpa_trend_7d", 0.0) or 0.0),
            "spend_trend_7d": float(row.get("spend_trend_7d", 0.0) or 0.0),
            "impression_share": float(row.get("impression_share", 0.0) or 0.0),
            "audience_quality_score": float(row.get("audience_quality_score", 0.5) or 0.5),
            "creative_fatigue_score": float(row.get("creative_fatigue_score", 0.0) or 0.0),
            "target_cpa": float(row.get("target_cpa", 50) or 50),
            "min_roas": float(row.get("min_roas", 1.0) or 1.0),
        },
        temporal={
            "hour": 12,
            "day_of_week": 0,
            "campaign_age_days": int(row.get("campaign_age_days", 0) or 0),
        },
        ml_features={
            "audience_quality": float(row.get("audience_quality_score", 0.5) or 0.5),
            "creative_fatigue": float(row.get("creative_fatigue_score", 0.0) or 0.0),
            "predicted_cvr": float(row.get("predicted_cvr", row.get("cvr", 0.02)) or 0.02),
            "predicted_ltv": float(row.get("predicted_ltv", 0.5) or 0.5),
            "propensity": float(row.get("propensity_score", 0.5) or 0.5),
        },
        platform=platform,
    )
    return state.to_tensor().numpy()


def create_from_dataframe(
    df: pd.DataFrame,
    output_path: str,
    include_platform_one_hot: bool = False,
) -> int:
    """
    Create cross-platform state vectors from a DataFrame.

    Args:
        df: DataFrame with campaign metrics
        output_path: Output CSV path
        include_platform_one_hot: Add platform one-hot features

    Returns:
        Number of state rows written
    """
    if _QA_IO_LOGGING:
        print(f"[IO] INPUT: df shape={df.shape}, columns={list(df.columns[:10])}...")
        print(f"[IO] INPUT: output_path={output_path}, include_platform_one_hot={include_platform_one_hot}")

    features = list(DEFAULT_FEATURES)
    if include_platform_one_hot:
        features = features + PLATFORM_ONE_HOT

    # Fill missing features with 0
    for f in features:
        if f not in df.columns:
            df = df.copy()
            df[f] = 0.0

    loader = BigQueryDataLoader(features=features)
    loader.fit_normalizer(df[features])
    states = loader.transform(df)
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

    out_df = df[["campaign_id", "created_at"]].copy() if "campaign_id" in df.columns and "created_at" in df.columns else pd.DataFrame()
    for i, feat in enumerate(loader.features):
        if i < states.shape[1]:
            out_df[f"state_{feat}" if i < 42 else feat] = states[:, i]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    norm_path = str(Path(output_path).with_suffix("")) + "_normalization.json"
    with open(norm_path, "w") as f:
        json.dump({"mean": loader._mean.tolist(), "std": loader._std.tolist(), "features": loader.features}, f, indent=2)

    if _QA_IO_LOGGING:
        print(f"[IO] OUTPUT: {output_path} | {len(out_df)} rows, state_dim={states.shape[1]}")
        print(f"[IO] OUTPUT: normalization_params={norm_path}")
        print(f"[IO] OUTPUT: Next: run_cross_platform_optimizer (--table or CSV), train_bigquery_offline")

    return len(out_df)


def create_from_bigquery(
    project_id: str,
    table: str,
    credentials_path: str = "",
    output_path: str = "Training_Data/outputs/cross_platform_states.csv",
    include_platform_one_hot: bool = True,
    limit: Optional[int] = None,
) -> int:
    """
    Fetch from BigQuery and create cross-platform state vectors.

    Args:
        project_id: GCP project ID
        table: BigQuery table (e.g. ad_metrics.campaign_states)
        credentials_path: Service account JSON path
        output_path: Output CSV path
        include_platform_one_hot: Add platform one-hot
        limit: Optional row limit

    Returns:
        Number of state rows written
    """
    creds_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    if _QA_IO_LOGGING:
        print(f"[IO] INPUT: project_id={project_id}, table={table}")
        print(f"[IO] INPUT: output_path={output_path}")

    loader = BigQueryDataLoader(
        project_id=project_id,
        table=table,
        credentials_path=creds_path,
        features=list(DEFAULT_FEATURES) + (PLATFORM_ONE_HOT if include_platform_one_hot else []),
    )
    df = loader.fetch_dataframe(limit=limit)
    states = loader.transform(df)
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

    out_df = df[["campaign_id", "created_at"]].copy() if "campaign_id" in df.columns and "created_at" in df.columns else pd.DataFrame()
    for i, feat in enumerate(loader.features):
        if i < states.shape[1]:
            out_df[feat] = states[:, i]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    norm_path = str(Path(output_path).with_suffix("")) + "_normalization.json"
    with open(norm_path, "w") as f:
        json.dump({"mean": loader._mean.tolist(), "std": loader._std.tolist(), "features": loader.features}, f, indent=2)

    if _QA_IO_LOGGING:
        print(f"[IO] OUTPUT: {output_path} | {len(out_df)} rows, state_dim={states.shape[1]}")
        print(f"[IO] OUTPUT: Next: run_cross_platform_optimizer, train_bigquery_offline")

    return len(out_df)


def main():
    parser = argparse.ArgumentParser(description="Create cross-platform state vectors")
    parser.add_argument("--input-csv", help="Input CSV with campaign metrics")
    parser.add_argument("--output", default="Training_Data/outputs/cross_platform_states.csv")
    parser.add_argument("--project-id", default=os.environ.get("DRL_BQ_PROJECT_ID", ""))
    parser.add_argument("--table", default="ad_metrics.campaign_states")
    parser.add_argument("--credentials", default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
    parser.add_argument("--platform-one-hot", action="store_true", default=True)
    parser.add_argument("--no-platform-one-hot", action="store_false", dest="platform_one_hot")
    parser.add_argument("--limit", type=int, help="Limit rows for BigQuery")

    args = parser.parse_args()

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        count = create_from_dataframe(
            df,
            args.output,
            include_platform_one_hot=args.platform_one_hot,
        )
    else:
        if not args.project_id:
            parser.error("--project-id or DRL_BQ_PROJECT_ID required for BigQuery")
        count = create_from_bigquery(
            project_id=args.project_id,
            table=args.table,
            credentials_path=args.credentials,
            output_path=args.output,
            include_platform_one_hot=args.platform_one_hot,
            limit=args.limit,
        )

    print(f"Created {count} cross-platform state rows -> {args.output}")


if __name__ == "__main__":
    main()
