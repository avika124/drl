"""
AI Optimization Engine — wires the DRL integration layer into the backend.

Exposes:
- drl_integration_layer: singleton DRLIntegrationLayer
- cross_platform_optimizer: singleton CrossPlatformOptimizer (None until init)
- get_drl_recommendations(): async helper used by the rest of the backend
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid heavy torch load at module-import time
_drl_integration_layer = None
_cross_platform_optimizer = None


def _ensure_drl_layer():
    global _drl_integration_layer
    if _drl_integration_layer is None:
        import sys
        # Ensure the DRL package is importable
        drl_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if drl_root not in sys.path:
            sys.path.insert(0, drl_root)

        from drl.drl_integration import DRLIntegrationLayer

        _drl_integration_layer = DRLIntegrationLayer(
            model_dir=os.getenv("DRL_MODEL_DIR", "models/drl"),
            device=os.getenv("DRL_DEVICE", "cpu"),
            min_confidence=float(os.getenv("DRL_MIN_CONFIDENCE", "0.7")),
            auto_apply_threshold=float(os.getenv("DRL_AUTO_APPLY_THRESHOLD", "0.85")),
        )
    return _drl_integration_layer


class _DRLProxy:
    """Thin proxy so ``drl_integration_layer`` can be imported at module level
    without triggering heavy DRL/torch imports until first attribute access."""

    def __getattr__(self, name):
        return getattr(_ensure_drl_layer(), name)

    def __repr__(self):
        return repr(_ensure_drl_layer())


drl_integration_layer = _DRLProxy()


class _CrossPlatformProxy:
    """Lazy proxy for cross-platform optimizer (created on first init)."""

    def __getattr__(self, name):
        global _cross_platform_optimizer
        if _cross_platform_optimizer is None:
            logger.warning("CrossPlatformOptimizer not yet initialised")
            return None
        return getattr(_cross_platform_optimizer, name)

    def __repr__(self):
        return repr(_cross_platform_optimizer)


cross_platform_optimizer = _CrossPlatformProxy()


# ── Public helpers ────────────────────────────────────────────────

async def get_drl_recommendations(
    campaign_id: str,
    campaign_data: Dict[str, Any],
    current_metrics: Dict[str, Any],
    optimization_goal: str = "roas",
    historical_metrics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """High-level entry point called by the rest of the backend.

    Falls back to a simple rule-based response when DRL is disabled via
    the ``DRL_ENABLED`` environment variable.
    """
    if os.getenv("DRL_ENABLED", "True").lower() != "true":
        return _rule_based_fallback(campaign_id, campaign_data, current_metrics)

    layer = _ensure_drl_layer()
    return await layer.get_optimization(
        campaign_id=campaign_id,
        campaign_data=campaign_data,
        metrics=current_metrics,
        optimization_goal=optimization_goal,
        historical_metrics=historical_metrics,
    )


def _rule_based_fallback(
    campaign_id: str,
    campaign_data: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Minimal rule-based recommendations when DRL is disabled."""
    recs = []
    roas = metrics.get("roas", 0.0)
    cpa = metrics.get("cpa", 0.0)

    if roas < 1.0:
        recs.append({
            "type": "budget_optimization",
            "action": "decrease_budget",
            "change_percent": -10.0,
            "rationale": "ROAS below 1.0 — reduce spend",
            "confidence": 0.6,
            "priority": 1,
        })
    if cpa > 100:
        recs.append({
            "type": "bid_optimization",
            "action": "decrease_bid",
            "change_percent": -15.0,
            "rationale": "CPA above $100 — lower bids",
            "confidence": 0.6,
            "priority": 1,
        })

    return {
        "campaign_id": campaign_id,
        "recommendations": recs,
        "confidence": 0.5,
        "requires_review": True,
        "auto_apply": False,
        "drl_enabled": False,
    }
