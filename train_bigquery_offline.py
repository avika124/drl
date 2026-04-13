"""
Offline training from BigQuery campaign state data.

This module:
- Pulls campaign state rows from BigQuery
- Builds pseudo-transitions by ordering states per campaign
- Infers continuous actions from CPC/spend velocity deltas
- Assigns discrete actions from simple heuristics
- Computes rewards using RewardComputer
- Trains SAC offline with CQL + PER
- Includes behavioral cloning pretrain step for actor warm-start

Configuration via environment variables:
  DRL_BQ_PROJECT_ID
  DRL_BQ_TABLE
  GOOGLE_APPLICATION_CREDENTIALS
  DRL_OUTPUT_DIR         (default: models/bq_run)
  DRL_ACTION_LOGS_PATH   (optional: path to synthetic action logs CSV)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .bigquery_loader import BigQueryDataLoader, DEFAULT_FEATURES
from .config import DRLConfig, TrainingConfig, OptimizationGoal
from .offline_trainer import OfflineTrainer
from .reward_functions import RewardComputer
from .replay_buffer import Transition
from .sac_agent import SACAgent


# Configuration from env vars (no hardcoded paths)
_PROJECT_ID = os.environ.get("DRL_BQ_PROJECT_ID", "")
_TABLE = os.environ.get("DRL_BQ_TABLE", "ad_metrics.campaign_states")
_OUTPUT_DIR = os.environ.get("DRL_OUTPUT_DIR", "models/bq_run")
_ACTION_LOGS_PATH = os.environ.get("DRL_ACTION_LOGS_PATH", "")

PLATFORM_ONE_HOT_FEATURES = [
    "is_google",
    "is_meta",
    "is_amazon",
    "is_tiktok",
    "is_linkedin",
]

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


def _infer_actions(current: pd.Series, nxt: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
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

    return (
        np.array([bid_adj, budget_adj], dtype=np.float32),
        np.array([audience_action, creative_action], dtype=np.int64),
    )


def _compute_reward(
    current: pd.Series,
    nxt: pd.Series,
    reward_computer: RewardComputer,
    reward_goal: OptimizationGoal = OptimizationGoal.ROAS,
    cpa_penalty_weight: float = 0.0,
    cpa_target_value: float = float("inf"),
) -> float:
    """Compute reward using RewardComputer."""
    clicks_before = float(current.get("clicks", 0.0) or 0.0)
    clicks_after = float(nxt.get("clicks", 0.0) or 0.0)
    cvr_before = float(current.get("cvr", 0.0) or 0.0)
    cvr_after = float(nxt.get("cvr", 0.0) or 0.0)
    cpc_before = float(current.get("cpc", 0.0) or 0.0)
    cpc_after = float(nxt.get("cpc", 0.0) or 0.0)

    conversions_before = round(clicks_before * cvr_before)
    conversions_after = round(clicks_after * cvr_after)
    spend_before = clicks_before * cpc_before
    spend_after = clicks_after * cpc_after
    roas_before = float(current.get("roas", 0.0) or 0.0)
    roas_after = float(nxt.get("roas", 0.0) or 0.0)
    revenue_before = spend_before * roas_before
    revenue_after = spend_after * roas_after

    metrics_before = {
        "roas": roas_before, "cpa": float(current.get("cpa", 0.0) or 0.0),
        "ctr": float(current.get("ctr", 0.0) or 0.0), "cvr": cvr_before,
        "conversions": conversions_before, "spend": spend_before, "revenue": revenue_before,
    }
    metrics_after = {
        "roas": roas_after, "cpa": float(nxt.get("cpa", 0.0) or 0.0),
        "ctr": float(nxt.get("ctr", 0.0) or 0.0), "cvr": cvr_after,
        "conversions": conversions_after, "spend": spend_after, "revenue": revenue_after,
    }

    reward = reward_computer.compute(
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        action={"bid_adjustment": 0.0, "budget_adjustment": 0.0},
        goal=reward_goal,
        constraints={"min_roas": 0.0, "target_cpa": cpa_target_value, "max_daily_spend": float("inf")},
        context={"hours_since_last_action": 24},
    )

    if cpa_penalty_weight > 0.0:
        cpa_after = float(metrics_after.get("cpa", 0.0) or 0.0)
        if np.isfinite(cpa_target_value):
            reward.total -= cpa_penalty_weight * max(0.0, cpa_after - cpa_target_value)
        else:
            reward.total -= cpa_penalty_weight * cpa_after

    scale = 1.0 / (1.0 + np.log1p(max(spend_after, 0.0)))
    return float(np.clip(reward.total * scale, -5.0, 5.0))


def build_transitions(
    credentials_path: str = "",
    table: str = "",
    include_platform_one_hot: bool = False,
    reward_goal: OptimizationGoal = OptimizationGoal.ROAS,
    cpa_penalty_weight: float = 0.0,
    cpa_target_value: float = float("inf"),
    output_dir: str = "",
) -> Tuple[List[Transition], Dict]:
    """
    Build transitions from BigQuery states.

    Returns:
        (transitions, normalization_params)
    """
    table = table or _TABLE
    output_dir = output_dir or _OUTPUT_DIR
    creds = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    loader_kwargs = {
        "project_id": _PROJECT_ID,
        "table": table,
        "credentials_path": creds,
    }
    if include_platform_one_hot:
        loader_kwargs["features"] = list(DEFAULT_FEATURES) + PLATFORM_ONE_HOT_FEATURES

    loader = BigQueryDataLoader(**loader_kwargs)
    state_df = loader.fetch_dataframe()
    states = loader.transform(state_df)
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    norm = {"mean": loader._mean.tolist(), "std": loader._std.tolist(), "features": loader.features}
    norm_path = os.path.join(output_dir, "normalization_params.json")
    with open(norm_path, "w") as f:
        json.dump(norm, f, indent=2)

    # Metadata for ordering + action inference
    try:
        from google.cloud import bigquery as bq_mod
        from google.oauth2 import service_account as sa_mod
        if creds:
            _creds = sa_mod.Credentials.from_service_account_file(creds)
            client = bq_mod.Client(project=_PROJECT_ID, credentials=_creds)
        else:
            client = bq_mod.Client(project=_PROJECT_ID or None)
        columns = ", ".join(META_COLUMNS)
        table_ref = f"{_PROJECT_ID}.{table}" if _PROJECT_ID else table
        query = f"SELECT {columns} FROM `{table_ref}`"
        meta_df = client.query(query).to_dataframe()
    except Exception:
        # Fall back: use loader dataframe columns that overlap with META_COLUMNS
        meta_df = state_df[[c for c in META_COLUMNS if c in state_df.columns]].copy()

    numeric_cols = meta_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if meta_df[col].isnull().any():
            meta_df[col] = meta_df[col].fillna(meta_df[col].median())
    meta_df = meta_df.fillna("")
    meta_df["row_idx"] = np.arange(len(meta_df))

    # Drop outliers
    for col, lo, hi in [
        ("ctr", 0.0, 0.5), ("cvr", 0.0, 0.5), ("cpc", 0.0, 100.0),
        ("cpa", 0.0, 1000.0), ("roas", 0.0, 50.0),
    ]:
        if col in meta_df.columns:
            meta_df = meta_df[(meta_df[col] >= lo) & (meta_df[col] <= hi)]

    reward_computer = RewardComputer()

    # Optional action log lookup
    action_lookup = {}
    action_logs_path = _ACTION_LOGS_PATH
    if action_logs_path and os.path.exists(action_logs_path):
        actions_df = pd.read_csv(action_logs_path)
        for _, row in actions_df.iterrows():
            key = (str(row["campaign_id"]), str(row["date"]))
            action_lookup[key] = {
                "bid_change_pct": float(row["bid_change_pct"]),
                "budget_change_pct": float(row["budget_change_pct"]),
                "audience_action_id": int(row["audience_action_id"]),
                "creative_action_id": int(row["creative_action_id"]),
            }

    transitions: List[Transition] = []
    if "campaign_id" not in meta_df.columns or "created_at" not in meta_df.columns:
        return transitions, norm

    for campaign_id, group in meta_df.sort_values("created_at").groupby("campaign_id"):
        if len(group) < 2:
            continue
        indices = group["row_idx"].to_numpy()
        for i in range(len(indices) - 1):
            idx_t = indices[i]
            idx_t1 = indices[i + 1]
            if idx_t >= len(states) or idx_t1 >= len(states):
                continue

            current = group.iloc[i]
            nxt = group.iloc[i + 1]

            date_key = str(pd.to_datetime(current.get("created_at")).date())
            lookup_key = (str(campaign_id), date_key)
            if lookup_key in action_lookup:
                action = action_lookup[lookup_key]
                cont_action = np.array([
                    np.clip(action["bid_change_pct"] / 100.0, -0.5, 0.5),
                    np.clip(action["budget_change_pct"] / 100.0, -0.3, 0.3),
                ], dtype=np.float32)
                disc_action = np.array([
                    action["audience_action_id"], action["creative_action_id"],
                ], dtype=np.int64)
            else:
                cont_action, disc_action = _infer_actions(current, nxt)

            reward = _compute_reward(
                current, nxt, reward_computer,
                reward_goal=reward_goal,
                cpa_penalty_weight=cpa_penalty_weight,
                cpa_target_value=cpa_target_value,
            )

            transitions.append(Transition(
                state=states[idx_t],
                continuous_action=cont_action,
                discrete_action=disc_action,
                reward=reward,
                next_state=states[idx_t1],
                done=(i == len(indices) - 2),
                campaign_id=str(campaign_id),
                timestamp=str(current.get("created_at", "")),
            ))

    return transitions, norm


def behavior_cloning_pretrain(
    agent: SACAgent,
    transitions: List[Transition],
    epochs: int = 5,
    batch_size: int = 256,
) -> None:
    """
    Supervised pretrain of actor to match inferred actions.
    Warm-starts the policy before CQL offline training for stability.
    """
    if not transitions:
        return

    states = torch.tensor(np.stack([t.state for t in transitions]), dtype=torch.float32)
    cont_actions = torch.tensor(
        np.stack([t.continuous_action for t in transitions]), dtype=torch.float32
    )
    disc_actions = torch.tensor(
        np.stack([t.discrete_action for t in transitions]), dtype=torch.long
    )

    dataset_size = states.shape[0]
    indices = np.arange(dataset_size)

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            batch_idx = indices[start: start + batch_size]
            batch_states = states[batch_idx]
            batch_cont = cont_actions[batch_idx]
            batch_disc = disc_actions[batch_idx]

            mean, _log_std, logits = agent.actor(batch_states)
            pred_cont = torch.tanh(mean)

            cont_loss = F.mse_loss(pred_cont, batch_cont)
            disc_loss = torch.tensor(0.0)
            for head_idx, head_logits in enumerate(logits):
                disc_loss = disc_loss + F.cross_entropy(
                    head_logits, batch_disc[:, head_idx]
                )

            loss = cont_loss + disc_loss

            agent.actor_optimizer.zero_grad()
            loss.backward()
            agent.actor_optimizer.step()


def train_from_bigquery(
    credentials_path: str = "",
    output_dir: str = "",
    bc_epochs: int = 5,
    train_epochs: int = 20,
    steps_per_epoch: int = 500,
) -> Dict:
    """
    Full training pipeline: BigQuery → transitions → BC pretrain → CQL.

    Returns:
        Training history dict.
    """
    output_dir = output_dir or _OUTPUT_DIR
    transitions, norm = build_transitions(
        credentials_path=credentials_path,
        output_dir=output_dir,
    )

    config = DRLConfig(state_dim=42, auto_entropy_tuning=False, alpha=0.2)
    train_cfg = TrainingConfig(
        batch_size=256, min_buffer_size=500,
        use_cql=True, cql_alpha=0.5, use_per=True,
    )
    agent = SACAgent(config, train_cfg, device="cpu")

    behavior_cloning_pretrain(agent, transitions, epochs=bc_epochs)

    trainer = OfflineTrainer(agent, train_cfg)
    trainer.load_transitions(transitions)

    history = trainer.train(
        num_epochs=train_epochs,
        steps_per_epoch=steps_per_epoch,
        checkpoint_dir=output_dir,
    )

    return {k: v[-1] if v else None for k, v in history.items()}
