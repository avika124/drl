"""
Generate synthetic action logs for offline DRL training.

===== STEP 1: P-MODEL TRAINING =====
[generate_synthetic_action_logs]
Description: Creates synthetic (campaign_id, date, bid_change_pct, budget_change_pct,
  audience_action_id, creative_action_id) tuples from campaign state data when real
  action logs are unavailable. Output CSV is consumed by train_bigquery_offline via
  DRL_ACTION_LOGS_PATH.

Input: BigQuery table or CSV with campaign states (campaign_id, created_at, cpc,
  spend_velocity, impressions, audience_quality_score, creative_fatigue_score, etc.)
Output: CSV with action logs -> train_bigquery_offline.build_transitions()
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# QA/Testing: Set True to enable input/output logging for traceability
_QA_IO_LOGGING = True

META_COLUMNS = [
    "campaign_id",
    "created_at",
    "impressions",
    "clicks",
    "cpc",
    "cvr",
    "ctr",
    "cpa",
    "roas",
    "spend_velocity",
    "spend_trend_7d",
    "impressions_trend_7d",
    "clicks_trend_7d",
    "audience_quality_score",
    "creative_fatigue_score",
    "impression_share",
]


def _infer_actions(current: pd.Series, nxt: pd.Series) -> tuple:
    """Infer continuous and discrete actions from state deltas."""
    cpc_before = float(current.get("cpc", 0.0) or 0.0)
    cpc_after = float(nxt.get("cpc", 0.0) or 0.0)
    bid_adj = (cpc_after - cpc_before) / cpc_before if cpc_before > 0 else 0.0

    impressions_before = float(current.get("impressions", 0.0) or 0.0)
    impressions_after = float(nxt.get("impressions", 0.0) or 0.0)
    volume_change = (
        (impressions_after - impressions_before) / impressions_before
        if impressions_before > 0 else 0.0
    )

    sv_before = float(current.get("spend_velocity", 0.0) or 0.0)
    sv_after = float(nxt.get("spend_velocity", 0.0) or 0.0)
    sv_change = (sv_after - sv_before) / sv_before if abs(sv_before) > 0 else 0.0

    budget_adj = 0.6 * volume_change + 0.4 * sv_change
    bid_adj = float(np.clip(bid_adj, -0.5, 0.5))
    budget_adj = float(np.clip(budget_adj, -0.3, 0.3))

    audience_action = 0
    creative_action = 0
    if float(current.get("audience_quality_score", 0.5)) < 0.4:
        audience_action = 2  # REFINE
    elif float(current.get("impression_share", 0.5)) < 0.3:
        audience_action = 1  # EXPAND
    if float(current.get("creative_fatigue_score", 0.0)) > 0.5:
        creative_action = 1  # ROTATE
    elif float(current.get("ctr", 0.0)) < 0.01:
        creative_action = 3  # TEST_NEW

    return bid_adj, budget_adj, audience_action, creative_action


def generate_from_dataframe(
    df: pd.DataFrame,
    output_path: str,
) -> int:
    """
    Generate synthetic action logs from a campaign state dataframe.

    Args:
        df: DataFrame with campaign_id, created_at, and metric columns
        output_path: Path to write CSV

    Returns:
        Number of action log rows generated
    """
    if _QA_IO_LOGGING:
        print(f"[IO] INPUT: df shape={df.shape}, columns={list(df.columns[:8])}...")
        print(f"[IO] INPUT: output_path={output_path}")

    if "campaign_id" not in df.columns:
        raise ValueError("DataFrame must have campaign_id column")
    if "created_at" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "created_at"})
    if "created_at" not in df.columns:
        raise ValueError("DataFrame must have created_at or date column")

    rows = []
    for campaign_id, group in df.sort_values("created_at").groupby("campaign_id"):
        if len(group) < 2:
            continue
        for i in range(len(group) - 1):
            current = group.iloc[i]
            nxt = group.iloc[i + 1]
            bid_pct, budget_pct, audience_action, creative_action = _infer_actions(
                current, nxt
            )
            date_str = str(pd.to_datetime(current.get("created_at")).date())
            rows.append({
                "campaign_id": str(campaign_id),
                "date": date_str,
                "bid_change_pct": bid_pct * 100,
                "budget_change_pct": budget_pct * 100,
                "audience_action_id": audience_action,
                "creative_action_id": creative_action,
            })

    out_df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    if _QA_IO_LOGGING:
        print(f"[IO] OUTPUT: {output_path} | {len(out_df)} rows")
        print(f"[IO] OUTPUT: Sample: campaign_id={out_df.iloc[0]['campaign_id']}, date={out_df.iloc[0]['date']}")
        print(f"[IO] OUTPUT: Next: train_bigquery_offline (set DRL_ACTION_LOGS_PATH={output_path})")

    return len(out_df)


def generate_from_bigquery(
    project_id: str,
    table: str,
    credentials_path: str = "",
    output_path: str = "Training_Data/outputs/synthetic_action_logs.csv",
    limit: Optional[int] = None,
) -> int:
    """
    Fetch campaign states from BigQuery and generate synthetic action logs.

    Args:
        project_id: GCP project ID
        table: BigQuery table (e.g. ad_metrics.campaign_states)
        credentials_path: Path to service account JSON
        output_path: Output CSV path
        limit: Optional row limit for sampling

    Returns:
        Number of action log rows generated
    """
    creds_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    if _QA_IO_LOGGING:
        print(f"[IO] INPUT: project_id={project_id}, table={table}")
        print(f"[IO] INPUT: credentials_path={creds_path or '(env)'}, output_path={output_path}")

    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account

        client = (
            bigquery.Client(
                project=project_id,
                credentials=service_account.Credentials.from_service_account_file(creds_path),
            )
            if creds_path
            else bigquery.Client(project=project_id)
        )
    except ImportError:
        raise ImportError("google-cloud-bigquery required. pip install google-cloud-bigquery")

    columns = ", ".join(META_COLUMNS)
    table_ref = f"{project_id}.{table}" if project_id else table
    query = f"SELECT {columns} FROM `{table_ref}`"
    if limit:
        query += f" LIMIT {int(limit)}"

    df = client.query(query).to_dataframe()
    return generate_from_dataframe(df, output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic action logs for DRL training")
    parser.add_argument("--input-csv", help="Input CSV with campaign states")
    parser.add_argument("--output", default="Training_Data/outputs/synthetic_action_logs.csv")
    parser.add_argument("--project-id", default=os.environ.get("DRL_BQ_PROJECT_ID", ""))
    parser.add_argument("--table", default="ad_metrics.campaign_states")
    parser.add_argument("--credentials", default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
    parser.add_argument("--limit", type=int, help="Limit rows for BigQuery")

    args = parser.parse_args()

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        count = generate_from_dataframe(df, args.output)
    else:
        if not args.project_id:
            parser.error("--project-id or DRL_BQ_PROJECT_ID required for BigQuery")
        count = generate_from_bigquery(
            project_id=args.project_id,
            table=args.table,
            credentials_path=args.credentials,
            output_path=args.output,
            limit=args.limit,
        )

    print(f"Generated {count} synthetic action log rows -> {args.output}")


if __name__ == "__main__":
    main()
