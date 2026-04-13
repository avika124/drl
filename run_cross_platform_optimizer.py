"""
Run cross-platform budget allocation + campaign optimization from BigQuery data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

from .config import GuardrailConfig
from .cross_platform_optimizer import CrossPlatformConfig, CrossPlatformOptimizer
from .hybrid_optimizer import HybridDRLLLMOptimizer
from .safe_agent import CampaignContext, SafeDRLAgent
from .sac_agent import load_sac_for_inference
from .state_action import CampaignState


def _bq_client(project_id: str, credentials_path: str) -> bigquery.Client:
    creds = service_account.Credentials.from_service_account_file(credentials_path)
    return bigquery.Client(project=project_id, credentials=creds)


def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    value = row.get(col, default)
    if pd.isna(value):
        return default
    return float(value)


def _safe_int(row: pd.Series, col: str, default: int = 0) -> int:
    value = row.get(col, default)
    if pd.isna(value):
        return default
    return int(value)


def _load_campaigns_from_bigquery(
    project_id: str,
    table: str,
    credentials_path: str,
    per_platform_limit: int,
) -> pd.DataFrame:
    # Step A: Load a balanced sample from each platform so allocation signals
    # are comparable across networks during a single run.
    client = _bq_client(project_id, credentials_path)
    query = f"""
    WITH ranked AS (
      SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY platform ORDER BY RAND()) AS rn
      FROM `{project_id}.{table}`
      WHERE platform IN ('google', 'meta', 'amazon', 'tiktok', 'linkedin')
    )
    SELECT * EXCEPT(rn)
    FROM ranked
    WHERE rn <= {int(per_platform_limit)}
    """
    return client.query(query).to_dataframe()


def _load_campaigns_from_sample_csv(sample_input_csv: str, per_platform_limit: int) -> pd.DataFrame:
    """
    Load local sample rows and adapt them into the schema used by the optimizer.
    """
    df = pd.read_csv(sample_input_csv).copy()
    if df.empty:
        return df

    # Resolve platform from explicit column or platform_encoding.
    platform_from_encoding = {
        0.0: "google",
        0.2: "meta",
        0.4: "amazon",
        0.6: "tiktok",
        0.8: "linkedin",
    }
    if "platform" not in df.columns:
        if "platform_encoding" in df.columns:
            df["platform"] = (
                df["platform_encoding"].round(1).map(platform_from_encoding).fillna("google")
            )
        else:
            default_platforms = ["google", "meta", "amazon", "tiktok", "linkedin"]
            df["platform"] = [default_platforms[i % len(default_platforms)] for i in range(len(df))]
    df["platform"] = df["platform"].astype(str).str.lower()

    # Add campaign_id if missing.
    if "campaign_id" not in df.columns:
        df["campaign_id"] = [f"sample_cmp_{i}" for i in range(len(df))]

    # Backfill clicks/impressions from normalized template columns.
    if "clicks" not in df.columns:
        if "click_volume" in df.columns:
            df["clicks"] = (df["click_volume"].astype(float) * 1000.0).round()
        else:
            df["clicks"] = 100.0
    if "impressions" not in df.columns:
        if "impression_volume" in df.columns:
            df["impressions"] = (df["impression_volume"].astype(float) * 10000.0).round()
        else:
            df["impressions"] = 1000.0

    # Optional fields used in _build_campaign_tuples.
    if "campaign_age_days" not in df.columns and "campaign_maturity" in df.columns:
        df["campaign_age_days"] = df["campaign_maturity"].astype(float) * 365.0
    if "is_holiday_season" not in df.columns and "is_holiday" in df.columns:
        df["is_holiday_season"] = df["is_holiday"]

    # Keep at most N rows per platform for parity with BigQuery mode.
    if per_platform_limit > 0:
        df = (
            df.assign(_rn=df.groupby("platform").cumcount() + 1)
            .loc[lambda x: x["_rn"] <= int(per_platform_limit)]
            .drop(columns=["_rn"])
            .reset_index(drop=True)
        )

    return df


def _build_campaign_tuples(df: pd.DataFrame) -> List[Tuple[CampaignState, CampaignContext, Dict[str, Any]]]:
    if df.empty:
        return []

    # Step B: Convert raw table rows into the tuple format expected by
    # CrossPlatformOptimizer:
    #   (CampaignState, CampaignContext, campaign_info_dict)
    campaigns = []
    max_impressions = max(float(df["impressions"].max()) if "impressions" in df.columns else 1.0, 1.0)
    max_clicks = max(float(df["clicks"].max()) if "clicks" in df.columns else 1.0, 1.0)

    platform_to_encoding = {
        "google": 0.0,
        "meta": 0.2,
        "amazon": 0.4,
        "tiktok": 0.6,
        "linkedin": 0.8,
    }

    for idx, row in df.reset_index(drop=True).iterrows():
        # Extract and sanitize core metrics.
        platform = str(row.get("platform", "google")).lower()
        campaign_id = str(row.get("campaign_id", f"cmp_{idx}"))
        clicks = max(0.0, _safe_float(row, "clicks", 0.0))
        impressions = max(0.0, _safe_float(row, "impressions", 0.0))
        cpc = max(0.01, _safe_float(row, "cpc", 0.01))
        cvr = max(0.0, _safe_float(row, "cvr", 0.0))
        roas = max(0.0, _safe_float(row, "roas", 0.0))
        cpa = max(0.0, _safe_float(row, "cpa", 0.0))
        spend = max(0.0, clicks * cpc)
        conversions = max(0, int(round(clicks * cvr)))
        revenue = spend * roas

        state = CampaignState(
            # State values are normalized/derived where needed to fit the
            # expected policy input space.
            campaign_id=campaign_id,
            platform=platform,
            ctr=_safe_float(row, "ctr", 0.0),
            cvr=cvr,
            roas=roas,
            cpa=cpa,
            cpc=cpc,
            cpm=_safe_float(row, "cpm", 0.0),
            spend_velocity=_safe_float(row, "spend_velocity", 0.0),
            impression_volume=impressions / max_impressions,
            click_volume=clicks / max_clicks,
            conversion_volume=min(1.0, conversions / max(max_clicks, 1.0)),
            # Keep raw hour/day scale aligned with BigQuery training data.
            hour_of_day=_safe_float(row, "hour_of_day", 12.0),
            day_of_week=_safe_float(row, "day_of_week", 3.0),
            day_of_month=0.5,
            is_weekend=_safe_float(row, "is_weekend", 0.0),
            is_holiday=_safe_float(row, "is_holiday_season", 0.0),
            days_remaining=0.5,
            ctr_trend_7d=_safe_float(row, "ctr_trend_7d", 0.0),
            cvr_trend_7d=_safe_float(row, "cvr_trend_7d", 0.0),
            roas_trend_7d=_safe_float(row, "roas_trend_7d", 0.0),
            cpa_trend_7d=_safe_float(row, "cpa_trend_7d", 0.0),
            spend_trend_7d=_safe_float(row, "spend_trend_7d", 0.0),
            impression_share=_safe_float(row, "impression_share", 0.0),
            auction_pressure=_safe_float(row, "auction_pressure", 0.0),
            competitive_position=_safe_float(row, "competitive_position", 0.0),
            audience_quality_score=_safe_float(row, "audience_quality_score", 0.0),
            creative_fatigue_score=_safe_float(row, "creative_fatigue_score", 0.0),
            predicted_cvr=_safe_float(row, "predicted_cvr", cvr),
            predicted_ltv=_safe_float(row, "predicted_ltv", 0.0),
            propensity_score=_safe_float(row, "propensity_score", 0.0),
            optimization_goal_encoding=0.0,
            platform_encoding=platform_to_encoding.get(platform, 0.0),
            campaign_maturity=min(1.0, _safe_float(row, "campaign_age_days", 0.0) / 365.0),
            budget_utilization=_safe_float(row, "budget_utilization", 0.0),
        )

        context = CampaignContext(
            # Context carries live constraints for guardrails validation.
            campaign_id=campaign_id,
            current_bid=cpc,
            current_budget=max(50.0, spend * 1.2),
            last_action_at=datetime.utcnow(),
            actions_today=0,
            current_roas=roas,
            current_cpa=cpa,
            target_cpa=max(0.01, cpa * 1.1) if cpa > 0 else None,
            min_roas=1.0,
            is_new_campaign=False,
            total_spend=spend,
        )

        info = {
            # campaign_info supports platform aggregation + rationale text.
            "platform": platform,
            "spend": spend,
            "revenue": revenue,
            "conversions": conversions,
            "clicks": int(round(clicks)),
            "impressions": int(round(impressions)),
            "campaign_name": str(row.get("campaign_name", campaign_id)),
        }
        campaigns.append((state, context, info))

    return campaigns


def _platform_metrics_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    # Step C: Build platform-level diagnostic metrics for easier review
    # of allocator behavior in the JSON output artifact.
    if df.empty or "platform" not in df.columns:
        return {}
    tmp = df.copy()
    if "clicks" in tmp.columns and "cpc" in tmp.columns:
        tmp["spend"] = tmp["clicks"].fillna(0.0) * tmp["cpc"].fillna(0.0)
    grouped = (
        tmp.groupby("platform", as_index=True)
        .agg(
            campaigns=("platform", "count"),
            avg_ctr=("ctr", "mean"),
            avg_cvr=("cvr", "mean"),
            avg_roas=("roas", "mean"),
            avg_cpc=("cpc", "mean"),
            total_spend=("spend", "sum"),
        )
        .sort_index()
    )
    out: Dict[str, Dict[str, float]] = {}
    for platform, row in grouped.iterrows():
        out[str(platform)] = {
            "campaigns": int(row["campaigns"]),
            "avg_ctr": float(row["avg_ctr"]),
            "avg_cvr": float(row["avg_cvr"]),
            "avg_roas": float(row["avg_roas"]),
            "avg_cpc": float(row["avg_cpc"]),
            "total_spend": float(row["total_spend"]),
        }
    return out


async def _run(args) -> Dict[str, Any]:
    # Step 1: Build the optimization stack.
    # SAC -> SafeDRLAgent (guardrails) -> HybridDRLLLMOptimizer -> CrossPlatformOptimizer
    state_dim = int(os.environ.get("DRL_STATE_DIM", "42"))
    sac, _ = load_sac_for_inference(
        model_dir=args.sac_model_dir,
        device="cpu",
        state_dim=state_dim,
    )
    safe_agent = SafeDRLAgent(agent=sac, guardrails=GuardrailConfig(), exploration_rate=0.05)
    hybrid = HybridDRLLLMOptimizer(drl_agent=safe_agent, enable_tactical=False)
    optimizer = CrossPlatformOptimizer(
        hybrid_optimizer=hybrid,
        config=CrossPlatformConfig(),
        max_concurrent_campaigns=10,
    )

    # Step 2: Load campaign rows (BigQuery mode or local sample mode).
    if args.sample:
        df = _load_campaigns_from_sample_csv(
            sample_input_csv=args.sample_input_csv,
            per_platform_limit=args.per_platform_limit,
        )
    else:
        df = _load_campaigns_from_bigquery(
            project_id=args.project_id,
            table=args.table,
            credentials_path=args.credentials,
            per_platform_limit=args.per_platform_limit,
        )
    campaigns = _build_campaign_tuples(df)
    if not campaigns:
        raise ValueError("No campaigns loaded for cross-platform optimization.")

    # Step 3: Decide portfolio budget.
    # If user does not provide --total-budget, infer it from observed spend.
    inferred_budget = float(sum(info.get("spend", 0.0) for _, _, info in campaigns))
    total_budget = args.total_budget if args.total_budget > 0 else max(inferred_budget, 1000.0)

    # Step 4: Run cross-platform allocation + per-campaign optimization.
    result = await optimizer.optimize_portfolio(
        organization_id=args.organization_id,
        campaigns=campaigns,
        total_budget=total_budget,
        force_rebalance=True,
    )

    # Step 5: Build final payload with both allocator output and diagnostics.
    payload = result.to_dict()
    payload["source_campaigns"] = len(campaigns)
    payload["source_table"] = (
        f"sample_csv:{args.sample_input_csv}"
        if args.sample
        else f"{args.project_id}.{args.table}"
    )
    payload["total_budget_used"] = total_budget
    payload["platform_metrics"] = _platform_metrics_summary(df)
    payload["mode"] = "sample" if args.sample else "bigquery"
    return payload


def main() -> None:
    # Step 0: CLI entrypoint. Parse runtime configuration.
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="ad-metrics-pipeline")
    parser.add_argument("--table", default="ad_metrics.cross_platform_states")
    parser.add_argument(
        "--credentials",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json"),
        help="Path to GCP service account JSON. Or set GOOGLE_APPLICATION_CREDENTIALS env var.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run using local sample CSV instead of BigQuery.",
    )
    parser.add_argument(
        "--sample-input-csv",
        default="Training_Data/input_template.csv",
        help="Sample input CSV used when --sample is set.",
    )
    parser.add_argument("--sac-model-dir", default="models/bq_run/final")
    parser.add_argument("--organization-id", default="org_cross_platform")
    parser.add_argument(
        "--total-budget",
        type=float,
        default=0.0,
        help="If <= 0, uses inferred current spend from loaded campaigns.",
    )
    parser.add_argument("--per-platform-limit", type=int, default=120)
    parser.add_argument(
        "--output-json",
        default="Training_Data/outputs/cross_platform/cross_platform_result.json",
    )
    args = parser.parse_args()

    # Execute async pipeline and write result artifact.
    payload = asyncio.run(_run(args))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote cross-platform result to {output_path}")
    print("Top-level summary:")
    print(
        json.dumps(
            {
                "portfolio_roas": payload.get("portfolio_roas"),
                "projected_portfolio_roas": payload.get("projected_portfolio_roas"),
                "allocation_confidence": payload.get("allocation_confidence"),
                "source_campaigns": payload.get("source_campaigns"),
                "source_table": payload.get("source_table"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
