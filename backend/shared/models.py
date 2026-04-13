"""
Shared SQLAlchemy models for the DRL advertising optimisation platform.

Provides:
- Base: declarative base for all models
- Campaign: core campaign entity
- DRLOptimizationAction: audit log of every DRL-driven action
"""

from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Campaign(Base):
    """Core campaign entity (minimal definition for FK reference)."""

    __tablename__ = "campaigns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    platform = Column(String(50), nullable=True)
    status = Column(String(20), default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    drl_actions = relationship(
        "DRLOptimizationAction",
        back_populates="campaign",
        lazy="dynamic",
    )


class DRLOptimizationAction(Base):
    """Audit log of every DRL-driven optimisation action."""

    __tablename__ = "drl_optimization_actions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    campaign_id = Column(
        UUID(as_uuid=True),
        ForeignKey("campaigns.id"),
        nullable=False,
        index=True,
    )
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    action_type = Column(String(50), nullable=False)
    action_details = Column(JSONB, nullable=True)
    state_before = Column(JSONB, nullable=True)
    metrics_before = Column(JSONB, nullable=True)
    metrics_after = Column(JSONB, nullable=True)

    reward = Column(Float, nullable=True)
    reward_breakdown = Column(JSONB, nullable=True)
    confidence = Column(Float, nullable=True)

    status = Column(String(20), default="pending")
    validation_result = Column(JSONB, nullable=True)
    requires_review = Column(Boolean, default=False)
    is_auto_applied = Column(Boolean, default=False)

    applied_at = Column(DateTime(timezone=True), nullable=True)
    outcome_observed_at = Column(DateTime(timezone=True), nullable=True)
    is_successful = Column(Boolean, nullable=True)
    reasoning = Column(Text, nullable=True)

    outcome_conversions = Column(Integer, nullable=True)
    outcome_revenue = Column(Float, nullable=True)
    outcome_spend = Column(Float, nullable=True)
    outcome_roas = Column(Float, nullable=True)

    campaign = relationship("Campaign", back_populates="drl_actions")
