"""
Run the DRL (SAC) policy on an input CSV and write an output CSV.

Input CSV: one row per campaign state with the 42 features in CampaignState order.
Output CSV: bid_adjustment, budget_adjustment, audience_action_id, creative_action_id
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
import torch

try:
    from .sac_agent import load_sac_for_inference
except ImportError:
    from drl.sac_agent import load_sac_for_inference


FEATURES = [
    "ctr",
    "cvr",
    "roas",
    "cpa",
    "cpc",
    "cpm",
    "spend_velocity",
    "impression_volume",
    "click_volume",
    "conversion_volume",
    "hour_of_day",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "is_holiday",
    "days_remaining",
    "ctr_trend_7d",
    "cvr_trend_7d",
    "roas_trend_7d",
    "cpa_trend_7d",
    "spend_trend_7d",
    "impression_share",
    "auction_pressure",
    "competitive_position",
    "audience_quality_score",
    "creative_fatigue_score",
    "predicted_cvr",
    "predicted_ltv",
    "propensity_score",
    "optimization_goal_encoding",
    "platform_encoding",
    "campaign_maturity",
    "budget_utilization",
    "log_daily_spend",
    "log_total_campaign_spend",
    "log_daily_budget",
    "segment_count",
    "top_segment_roas",
    "avg_frequency",
    "target_cpa_norm",
    "min_roas_norm",
    "daily_budget_limit_norm",
]


PLATFORMS = ["google", "meta", "amazon", "tiktok", "linkedin"]
PLATFORM_ENCODING_MAP = {
    0.0: "google",
    0.2: "meta",
    0.4: "amazon",
    0.6: "tiktok",
    0.8: "linkedin",
}


def _build_feature_frame(df: pd.DataFrame, model_features):
    work_df = df.copy(deep=True)

    # Backfill one-hot platform columns if model expects them.
    for p in PLATFORMS:
        col = f"is_{p}"
        if col in model_features and col not in work_df.columns:
            if "platform" in work_df.columns:
                work_df[col] = (work_df["platform"].str.lower() == p).astype(int)
            elif "platform_encoding" in work_df.columns:
                inferred_platform = (
                    work_df["platform_encoding"].round(1).map(PLATFORM_ENCODING_MAP).fillna("google")
                )
                work_df[col] = (inferred_platform == p).astype(int)
            else:
                work_df[col] = 0

    # Derive BigQuery-style columns from template-style columns when needed.
    derived_defaults = {
        "impressions": ("impression_volume", 10000.0),
        "clicks": ("click_volume", 1000.0),
        "impressions_trend_7d": ("spend_trend_7d", 1.0),
        "clicks_trend_7d": ("ctr_trend_7d", 1.0),
        "conversions_trend_7d": ("cvr_trend_7d", 1.0),
        "campaign_age_days": ("campaign_maturity", 365.0),
        "is_holiday_season": ("is_holiday", 1.0),
    }
    for target_col, (src_col, scale) in derived_defaults.items():
        if target_col in model_features and target_col not in work_df.columns and src_col in work_df.columns:
            work_df[target_col] = work_df[src_col].astype(float) * scale

    if "predicted_revenue" in model_features and "predicted_revenue" not in work_df.columns:
        if {"predicted_ltv", "clicks", "cvr"}.issubset(set(work_df.columns)):
            work_df["predicted_revenue"] = (
                work_df["predicted_ltv"].astype(float)
                * work_df["clicks"].astype(float)
                * work_df["cvr"].astype(float)
            )
        elif {"roas", "cpc", "clicks"}.issubset(set(work_df.columns)):
            work_df["predicted_revenue"] = (
                work_df["roas"].astype(float)
                * work_df["cpc"].astype(float)
                * work_df["clicks"].astype(float)
            )

    if "engagement_score" in model_features and "engagement_score" not in work_df.columns and "ctr" in work_df.columns:
        work_df["engagement_score"] = (work_df["ctr"].astype(float) * 10.0).clip(lower=0.0, upper=1.0)

    for col in model_features:
        if col not in work_df.columns:
            work_df[col] = 0.0

    return work_df[model_features].astype(float)


def _apply_normalization(feature_df: pd.DataFrame, model_features, normalization_path: str | None):
    if not normalization_path:
        return feature_df

    try:
        with open(normalization_path, "r", encoding="utf-8") as f:
            norm = json.load(f)
    except Exception:
        return feature_df

    if norm.get("features", []) != list(model_features):
        return feature_df

    out = feature_df.copy(deep=True)
    means = norm.get("mean", [])
    stds = norm.get("std", [])
    for i, col in enumerate(model_features):
        mean = float(means[i]) if i < len(means) else 0.0
        std = float(stds[i]) if i < len(stds) else 1.0
        if std == 0.0:
            std = 1.0
        out[col] = (out[col] - mean) / std
    return out


def _apply_business_guardrails(out: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional rule layer to avoid clearly counter-intuitive actions in demos.
    """
    guarded = out.copy(deep=True)
    if "cpa" in source_df.columns:
        high_cpa = source_df["cpa"].astype(float) >= 1.0
        # Force visibly defensive actions for clearly bad CPA rows.
        guarded.loc[high_cpa, "bid_adjustment"] = np.minimum(
            guarded.loc[high_cpa, "bid_adjustment"], -0.15
        )
        guarded.loc[high_cpa, "budget_adjustment"] = np.minimum(
            guarded.loc[high_cpa, "budget_adjustment"], -0.20
        )
        guarded.loc[high_cpa, "audience_action_id"] = guarded.loc[high_cpa, "audience_action_id"].clip(lower=2)
        guarded.loc[high_cpa, "creative_action_id"] = guarded.loc[high_cpa, "creative_action_id"].clip(lower=1)
    if "roas" in source_df.columns:
        low_roas = source_df["roas"].astype(float) <= 1.0
        guarded.loc[low_roas, "bid_adjustment"] = np.minimum(
            guarded.loc[low_roas, "bid_adjustment"], -0.10
        )
        guarded.loc[low_roas, "budget_adjustment"] = np.minimum(
            guarded.loc[low_roas, "budget_adjustment"], -0.15
        )

    # Monotonic stress penalty: progressively push actions more defensive as
    # CPA worsens and/or ROAS falls, even before hard thresholds are hit.
    if "cpa" in source_df.columns and "roas" in source_df.columns:
        cpa = source_df["cpa"].astype(float)
        roas = source_df["roas"].astype(float)
        cpa_stress = np.maximum(0.0, cpa - 0.20) * 0.20
        roas_stress = np.maximum(0.0, 1.00 - roas) * 0.30
        total_stress = np.minimum(0.35, cpa_stress + roas_stress)
        guarded["bid_adjustment"] = guarded["bid_adjustment"] - total_stress
        guarded["budget_adjustment"] = guarded["budget_adjustment"] - np.minimum(0.40, total_stress * 1.3)

    # Keep outputs within practical action bounds.
    guarded["bid_adjustment"] = guarded["bid_adjustment"].clip(-0.50, 0.50)
    guarded["budget_adjustment"] = guarded["budget_adjustment"].clip(-0.30, 0.30)
    return guarded


def run(
    input_csv: str,
    output_csv: str,
    model_dir: str,
    show: bool = False,
    deterministic: bool = True,
    normalization_path: str | None = None,
    apply_business_guardrails: bool = False,
    segments=None,
):
    df = pd.read_csv(input_csv)

    import os
    state_dim = int(os.environ.get("DRL_STATE_DIM", "42"))
    agent, ckpt_features = load_sac_for_inference(
        model_dir=model_dir,
        device="cpu",
        state_dim=state_dim,
    )
    model_features = ckpt_features if ckpt_features else FEATURES

    feature_df = _build_feature_frame(df, model_features)
    feature_df = _apply_normalization(feature_df, model_features, normalization_path)
    states = torch.tensor(feature_df.to_numpy(), dtype=torch.float32)

    with torch.no_grad():
        if deterministic:
            mean, _log_std, logits = agent.actor(states)
            cont = torch.tanh(mean)
            disc_idx = torch.stack([torch.argmax(head_logits, dim=1) for head_logits in logits], dim=1)
        else:
            cont, _disc_soft, disc_idx, _logp, _ent = agent.actor.sample(states)

    out = pd.DataFrame({
        "bid_adjustment": cont[:, 0].detach().cpu().numpy(),
        "budget_adjustment": cont[:, 1].detach().cpu().numpy(),
        "audience_action_id": disc_idx[:, 0].detach().cpu().numpy(),
        "creative_action_id": disc_idx[:, 1].detach().cpu().numpy(),
    })
    for col in ["roas", "cpa", "platform", "platform_encoding"]:
        if col in df.columns:
            out[col] = df[col].values
    if apply_business_guardrails:
        out = _apply_business_guardrails(out, df)

    if segments:
        from .audience_constraints import AudienceConstraintManager
        from .state_action import ActionSpace
        mgr = AudienceConstraintManager(segments)
        for idx, row in out.iterrows():
            action = ActionSpace(audience_action=int(row.get("audience_action_id", 0)))
            result = mgr.allocate_budget(
                platform_budget=1000.0,
                action=action,
                performance_signals={},
            )
            if show:
                print(f"\n-- Row {idx} segment allocations --")
                for a in result.allocations:
                    print(f"   {a.segment_id}: {a.recommended_budget_pct:.1%}")

    out.to_csv(output_csv, index=False)
    if show:
        print("Wrote:", output_csv)
        print(out.head())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-dir",
        default="models/bq_run/final",
        help="Directory containing SAC agent.pt checkpoint.",
    )
    parser.add_argument("--stochastic", action="store_true", help="Use sampled policy outputs instead of deterministic outputs.")
    parser.add_argument(
        "--normalization",
        default="",
        help="Path to normalization params json. Set empty string to disable.",
    )
    parser.add_argument("--business-guardrails", action="store_true", help="Apply simple CPA/ROAS sanity rules to final actions.")
    parser.add_argument("--segments", default=None, help="Path to audience segments JSON file.")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    segments = None
    if args.segments:
        from .audience_constraints import AudienceSegment
        with open(args.segments, "r", encoding="utf-8") as _f:
            import json as _json
            segments = [AudienceSegment(**s) for s in _json.load(_f)]

    run(
        args.input,
        args.output,
        args.model_dir,
        show=args.show,
        deterministic=not args.stochastic,
        normalization_path=(args.normalization or None),
        apply_business_guardrails=args.business_guardrails,
        segments=segments,
    )


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Consistency guarantee
# ---------------------------------------------------------------------------
# Both single-platform and cross-platform entry points now share the same
# SAC model via load_sac_for_inference (sac_agent.py).
#
# For any given input row and the same SAC checkpoint:
#   python -m drl.run_drl_on_csv --input in.csv --model-dir models/bq_run/final --output a.csv
# must produce the same bid_adjustment, budget_adjustment, audience_action_id,
# and creative_action_id as:
#   CrossPlatformOptimizer called with a single-campaign portfolio from in.csv
# because the single-platform passthrough in cross_platform_optimizer.py
# bypasses allocation and routes directly to the same DRL actor.
# ---------------------------------------------------------------------------
