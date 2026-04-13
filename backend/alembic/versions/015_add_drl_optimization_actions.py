"""add drl_optimization_actions table

Revision ID: 015
Revises: 014
Create Date: 2026-02-19
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "drl_optimization_actions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("campaign_id", UUID(as_uuid=True), sa.ForeignKey("campaigns.id"), nullable=False),
        sa.Column("organization_id", UUID(as_uuid=True), nullable=False),
        sa.Column("platform", sa.String(50), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("action_type", sa.String(50), nullable=False),
        sa.Column("action_details", JSONB),
        sa.Column("state_before", JSONB),
        sa.Column("metrics_before", JSONB),
        sa.Column("metrics_after", JSONB),
        sa.Column("reward", sa.Float),
        sa.Column("reward_breakdown", JSONB),
        sa.Column("confidence", sa.Float),
        sa.Column("model_version", sa.String(50)),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("validation_result", JSONB),
        sa.Column("requires_review", sa.Boolean, server_default="false"),
        sa.Column("is_auto_applied", sa.Boolean, server_default="false"),
        sa.Column("applied_at", sa.DateTime(timezone=True)),
        sa.Column("outcome_observed_at", sa.DateTime(timezone=True)),
        sa.Column("outcome_recorded", sa.Boolean, server_default="false"),
        sa.Column("is_successful", sa.Boolean),
        sa.Column("reasoning", sa.Text),
        sa.Column("outcome_conversions", sa.Integer),
        sa.Column("outcome_revenue", sa.Float),
        sa.Column("outcome_spend", sa.Float),
        sa.Column("outcome_roas", sa.Float),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_drl_actions_campaign_id", "drl_optimization_actions", ["campaign_id"])
    op.create_index("ix_drl_actions_org_id", "drl_optimization_actions", ["organization_id"])
    op.create_index("ix_drl_actions_timestamp", "drl_optimization_actions", ["timestamp"])
    op.create_index("ix_drl_actions_campaign_ts", "drl_optimization_actions", ["campaign_id", "timestamp"])
    op.create_index("ix_drl_actions_outcome", "drl_optimization_actions", ["outcome_recorded", "timestamp"])
    op.create_index("ix_drl_actions_platform", "drl_optimization_actions", ["platform"])


def downgrade() -> None:
    op.drop_table("drl_optimization_actions")
