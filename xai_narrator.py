"""
Explainable AI (xAI) Narrative Layer for DRL Optimization.

Provides:
- OptimizationNarrator: generates plain-English run narratives
  - Campaign-level narratives (P-Model decisions)
  - Portfolio-level narratives (X-Model allocation decisions)
- ParameterGlossary: look-up definitions for every key metric/parameter
- RunNarrative: structured container for a single run's explanation
- PortfolioNarrative: structured container for X-Model allocation explanations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .state_action import (
    AudienceAction,
    CampaignState,
    ActionSpace,
    CreativeAction,
    DRLDirective,
)


# ── Data containers ───────────────────────────────────────────────

@dataclass
class RunNarrative:
    situation_summary: str
    decision_summary: str
    reasoning: List[str]
    confidence_explanation: str
    reasonability_check: str
    full_narrative: str


@dataclass
class PortfolioNarrative:
    """Structured explanation for an X-Model portfolio allocation decision."""
    portfolio_summary: str
    allocation_decision: str
    platform_reasoning: List[str]
    confidence_explanation: str
    risk_assessment: str
    full_narrative: str


# ── Narrator ──────────────────────────────────────────────────────

class OptimizationNarrator:
    """Converts raw DRL inputs/outputs into a plain-English narrative."""

    # ── public entry point ────────────────────────────────────────

    def generate_run_narrative(
        self,
        state: CampaignState,
        action: ActionSpace,
        directive: DRLDirective,
        recommendations: list,
        reward_breakdown: Optional[dict] = None,
    ) -> RunNarrative:
        situation = self._describe_state_health(state)
        decision = self._describe_decision(action, directive)
        reasoning = self._explain_action(action, directive, state)
        confidence = self._explain_confidence(action, state)
        reasonability = self._check_reasonability(action)

        full = (
            f"=== Situation ===\n{situation}\n\n"
            f"=== Decision ===\n{decision}\n\n"
            f"=== Reasoning ===\n"
            + "\n".join(f"  {b}" for b in reasoning)
            + f"\n\n=== Confidence ===\n{confidence}\n\n"
            f"=== Reasonability ===\n{reasonability}"
        )

        return RunNarrative(
            situation_summary=situation,
            decision_summary=decision,
            reasoning=reasoning,
            confidence_explanation=confidence,
            reasonability_check=reasonability,
            full_narrative=full,
        )

    # ── situation ─────────────────────────────────────────────────

    def _describe_state_health(self, state: CampaignState) -> str:
        signals: List[str] = []

        if state.roas_trend_7d > 0.10:
            signals.append(f"a strong positive 7-day ROAS trend of +{state.roas_trend_7d:.0%}")
        elif state.roas_trend_7d < -0.10:
            signals.append(f"a declining 7-day ROAS trend of {state.roas_trend_7d:.0%}")

        if state.creative_fatigue_score > 0.65:
            signals.append(f"elevated creative fatigue at {state.creative_fatigue_score:.2f}")

        if state.budget_utilization > 0.80:
            signals.append(f"high budget utilisation at {state.budget_utilization:.0%}")
        elif state.budget_utilization < 0.30:
            signals.append(f"low budget utilisation at {state.budget_utilization:.0%}")

        if state.ctr_trend_7d < -0.10:
            signals.append(f"a declining CTR trend of {state.ctr_trend_7d:.0%}")
        elif state.ctr_trend_7d > 0.10:
            signals.append(f"a positive CTR trend of +{state.ctr_trend_7d:.0%}")

        if state.audience_quality_score > 0.75:
            signals.append("a high-quality audience score")

        if state.log_daily_spend < 0.10:
            signals.append("very low absolute daily spend")

        if state.impression_share > 0.70:
            signals.append(f"strong impression share at {state.impression_share:.0%}")
        elif state.impression_share < 0.20:
            signals.append(f"low impression share at {state.impression_share:.0%}")

        top = signals[:3] if signals else ["no strongly notable signals"]

        if len(top) == 1:
            body = f"The campaign is showing {top[0]}."
        elif len(top) == 2:
            body = f"The campaign is showing {top[0]} and {top[1]}."
        else:
            body = (
                f"The campaign is spending {state.budget_utilization:.0%} of its "
                f"daily budget and showing {top[0]}, which signals "
                f"{'healthy momentum' if state.roas_trend_7d > 0 else 'potential concern'}. "
                f"Additionally, it shows {top[1]} and {top[2]}."
            )

        return body

    # ── decision summary ──────────────────────────────────────────

    def _describe_decision(self, action: ActionSpace, directive: DRLDirective) -> str:
        parts: List[str] = []

        if action.bid_adjustment > 0.05:
            parts.append(f"increasing the bid by {action.bid_adjustment:.0%}")
        elif action.bid_adjustment < -0.05:
            parts.append(f"decreasing the bid by {abs(action.bid_adjustment):.0%}")

        if action.budget_adjustment > 0.05:
            parts.append(f"scaling the daily budget up by {action.budget_adjustment:.0%}")
        elif action.budget_adjustment < -0.05:
            parts.append(f"reducing the daily budget by {abs(action.budget_adjustment):.0%}")

        aud = AudienceAction(action.audience_action)
        if aud != AudienceAction.HOLD:
            parts.append(f"{aud.name.lower().replace('_', ' ')}ing the audience")

        cre = CreativeAction(action.creative_action)
        if cre == CreativeAction.ROTATE:
            parts.append("rotating creatives")
        elif cre == CreativeAction.PAUSE_UNDERPERFORMING:
            parts.append("pausing underperforming creatives")
        elif cre == CreativeAction.TEST_NEW:
            parts.append("testing new creative variants")

        if not parts:
            return "The model recommends holding all current settings — no significant action needed."

        joined = ", ".join(parts[:-1]) + (" and " + parts[-1] if len(parts) > 1 else parts[0])
        return f"The model recommends {joined}."

    # ── reasoning bullets ─────────────────────────────────────────

    def _explain_action(
        self,
        action: ActionSpace,
        directive: DRLDirective,
        state: CampaignState,
    ) -> List[str]:
        bullets: List[str] = []

        # Bid
        if action.bid_adjustment > 0.10:
            bullets.append(
                f"Impression share {state.impression_share:.0%} "
                f"→ increase bid +{action.bid_adjustment:.0%} "
                "because capturing more impression share accelerates reach."
            )
        elif action.bid_adjustment < -0.10:
            bullets.append(
                f"CPA pressure (state cpa={state.cpa:.2f}) "
                f"→ decrease bid {action.bid_adjustment:.0%} "
                "because reducing bids improves cost efficiency."
            )

        # Budget
        if action.budget_adjustment > 0.10:
            trigger = (
                f"ROAS trend +{state.roas_trend_7d:.0%}" if state.roas_trend_7d > 0.05
                else f"budget utilisation {state.budget_utilization:.0%}"
            )
            bullets.append(
                f"{trigger} → scale budget +{action.budget_adjustment:.0%} "
                "because positive momentum signals room to scale spend."
            )
        elif action.budget_adjustment < -0.10:
            bullets.append(
                f"Diminishing returns detected "
                f"→ reduce budget {action.budget_adjustment:.0%} "
                "because marginal ROAS is declining at current spend level."
            )

        # Audience
        aud = AudienceAction(action.audience_action)
        if aud == AudienceAction.EXPAND:
            trigger = (
                "audience quality score above threshold"
                if state.audience_quality_score > 0.5
                else "positive ROAS trend"
            )
            bullets.append(
                f"{trigger} → audience: EXPAND "
                "because expanding reach distributes impressions more efficiently."
            )
        elif aud == AudienceAction.REFINE:
            bullets.append(
                "CVR or auction pressure elevated → audience: REFINE "
                "because tightening targeting improves conversion efficiency."
            )
        elif aud == AudienceAction.EXCLUDE:
            bullets.append(
                "Conversion volume declining and CPA rising → audience: EXCLUDE "
                "because removing poor segments reduces wasted spend."
            )

        # Creative
        cre = CreativeAction(action.creative_action)
        if cre == CreativeAction.ROTATE:
            bullets.append(
                f"Creative fatigue {state.creative_fatigue_score:.2f} "
                "→ rotate creatives because scores above 0.65 predict a CTR drop within 48 hours."
            )
        elif cre == CreativeAction.PAUSE_UNDERPERFORMING:
            bullets.append(
                "CTR declining and fatigue high → pause underperforming creatives "
                "to concentrate spend on top-performing ads."
            )
        elif cre == CreativeAction.TEST_NEW:
            bullets.append(
                "ROAS plateau detected → test new creatives "
                "to discover fresh messaging that can break through the plateau."
            )

        if not bullets:
            bullets.append("All signals are neutral — holding current settings.")

        return [f"\u2022 {b}" for b in bullets]

    # ── confidence ────────────────────────────────────────────────

    @staticmethod
    def _explain_confidence(action: ActionSpace, state: CampaignState) -> str:
        pct = action.confidence * 100
        if action.confidence >= 0.85:
            qualifier = "highly"
        elif action.confidence >= 0.70:
            qualifier = "reasonably"
        elif action.confidence >= 0.50:
            qualifier = "moderately"
        else:
            qualifier = "only marginally"

        maturity = "a mature" if state.campaign_maturity > 0.3 else "a young"
        return (
            f"The model is {qualifier} confident at {pct:.0f}% "
            f"because the state signals are {'consistent' if action.confidence >= 0.7 else 'mixed'} "
            f"and this is {maturity} campaign (maturity {state.campaign_maturity:.2f})."
        )

    # ── portfolio narrative (X-Model) ─────────────────────────────

    def generate_portfolio_narrative(
        self,
        x_state_dict: Dict[str, Any],
        allocation_weights: Dict[str, float],
        previous_weights: Optional[Dict[str, float]] = None,
        confidence: float = 0.0,
        q_value: float = 0.0,
    ) -> PortfolioNarrative:
        """
        Generate a plain-English narrative for an X-Model allocation decision.

        Args:
            x_state_dict: XModelState.to_dict() — portfolio observation.
            allocation_weights: Platform → share (sum to 1.0).
            previous_weights: Previous allocation for shift context.
            confidence: X-Model confidence (0–1).
            q_value: Estimated Q-value of the allocation.
        """
        portfolio_summary = self._describe_portfolio(x_state_dict)
        allocation_decision = self._describe_allocation(
            allocation_weights, previous_weights
        )
        platform_reasoning = self._explain_allocation(
            x_state_dict, allocation_weights, previous_weights
        )
        conf_explanation = self._explain_portfolio_confidence(
            confidence, q_value, allocation_weights
        )
        risk = self._assess_portfolio_risk(
            allocation_weights, previous_weights, x_state_dict
        )

        full = (
            f"=== Portfolio Situation ===\n{portfolio_summary}\n\n"
            f"=== Allocation Decision ===\n{allocation_decision}\n\n"
            f"=== Platform Reasoning ===\n"
            + "\n".join(f"  {b}" for b in platform_reasoning)
            + f"\n\n=== Confidence ===\n{conf_explanation}\n\n"
            f"=== Risk Assessment ===\n{risk}"
        )

        return PortfolioNarrative(
            portfolio_summary=portfolio_summary,
            allocation_decision=allocation_decision,
            platform_reasoning=platform_reasoning,
            confidence_explanation=conf_explanation,
            risk_assessment=risk,
            full_narrative=full,
        )

    def _describe_portfolio(self, x_state_dict: Dict[str, Any]) -> str:
        """Describe the current portfolio health."""
        pf = x_state_dict.get("platform_features", {})
        roas = x_state_dict.get("portfolio_roas", 0.0)
        util = x_state_dict.get("budget_utilization", 0.0)
        hhi = x_state_dict.get("portfolio_hhi", 0.0)
        active_ratio = x_state_dict.get("active_platform_ratio", 0.0)

        # De-normalize portfolio ROAS (was divided by 5.0 in build_x_state)
        display_roas = roas * 5.0

        signals: List[str] = []
        if display_roas > 3.0:
            signals.append(f"strong portfolio ROAS of {display_roas:.1f}x")
        elif display_roas < 1.5:
            signals.append(f"below-target portfolio ROAS of {display_roas:.1f}x")

        if hhi > 0.35:
            signals.append("high budget concentration (HHI {:.2f})".format(hhi))
        elif hhi < 0.20 and active_ratio >= 0.6:
            signals.append("well-diversified budget allocation")

        if util > 0.85:
            signals.append(f"high budget utilisation at {util:.0%}")
        elif util < 0.40:
            signals.append(f"low budget utilisation at {util:.0%}")

        # Check for platforms with strong/weak trends
        strong = []
        weak = []
        for plat, feats in pf.items():
            trend = feats.get("roas_trend_7d", 0.0)
            if trend > 0.10:
                strong.append(plat)
            elif trend < -0.10:
                weak.append(plat)

        if strong:
            signals.append(f"positive ROAS trends on {', '.join(strong)}")
        if weak:
            signals.append(f"declining ROAS trends on {', '.join(weak)}")

        if not signals:
            return "The portfolio is in a neutral state with no strongly notable signals."

        body = "The portfolio is showing " + ", ".join(signals[:3]) + "."
        return body

    def _describe_allocation(
        self,
        weights: Dict[str, float],
        previous: Optional[Dict[str, float]],
    ) -> str:
        """Summarise the allocation decision."""
        sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        parts = [f"{p} {w:.0%}" for p, w in sorted_w]
        summary = "The X-Model allocates budget as: " + ", ".join(parts) + "."

        if previous:
            shifts: List[str] = []
            for plat, w in weights.items():
                prev_w = previous.get(plat, 0.0)
                delta = w - prev_w
                if abs(delta) >= 0.03:
                    direction = "+" if delta > 0 else ""
                    shifts.append(f"{plat} {direction}{delta:.0%}")
            if shifts:
                summary += " Shifts from previous: " + ", ".join(shifts) + "."
            else:
                summary += " No significant shifts from previous allocation."

        return summary

    def _explain_allocation(
        self,
        x_state_dict: Dict[str, Any],
        weights: Dict[str, float],
        previous: Optional[Dict[str, float]],
    ) -> List[str]:
        """Generate per-platform reasoning bullets."""
        pf = x_state_dict.get("platform_features", {})
        bullets: List[str] = []

        for plat, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            feats = pf.get(plat, {})
            roas = feats.get("roas", 0.0) * 5.0  # de-normalize
            m_roas = feats.get("marginal_roas", 0.0) * 5.0
            trend = feats.get("roas_trend_7d", 0.0)
            share = feats.get("spend_share", 0.0)

            prev_w = (previous or {}).get(plat, w)
            delta = w - prev_w

            reasons: List[str] = []
            if roas > 2.5:
                reasons.append(f"strong ROAS ({roas:.1f}x)")
            elif roas < 1.0:
                reasons.append(f"weak ROAS ({roas:.1f}x)")

            if m_roas > 2.0:
                reasons.append(f"high marginal ROAS ({m_roas:.1f}x)")
            elif m_roas < 0.8:
                reasons.append("diminishing marginal returns")

            if trend > 0.10:
                reasons.append("positive 7-day trend")
            elif trend < -0.10:
                reasons.append("declining 7-day trend")

            if abs(delta) >= 0.03:
                direction = "increase" if delta > 0 else "decrease"
                reason_str = ", ".join(reasons) if reasons else "portfolio rebalancing"
                bullets.append(
                    f"{plat.capitalize()}: {direction} to {w:.0%} "
                    f"(was {prev_w:.0%}) because {reason_str}."
                )
            elif reasons:
                bullets.append(
                    f"{plat.capitalize()}: hold at {w:.0%} — {', '.join(reasons)}."
                )

        if not bullets:
            bullets.append("All platforms held at current allocations — signals are neutral.")

        return [f"\u2022 {b}" for b in bullets]

    @staticmethod
    def _explain_portfolio_confidence(
        confidence: float, q_value: float, weights: Dict[str, float],
    ) -> str:
        """Explain confidence in the portfolio allocation."""
        pct = confidence * 100

        if confidence >= 0.85:
            qualifier = "highly"
        elif confidence >= 0.70:
            qualifier = "reasonably"
        elif confidence >= 0.50:
            qualifier = "moderately"
        else:
            qualifier = "only marginally"

        n_platforms = sum(1 for w in weights.values() if w >= 0.05)
        return (
            f"The X-Model is {qualifier} confident at {pct:.0f}% in this allocation "
            f"across {n_platforms} active platform(s) "
            f"(estimated Q-value: {q_value:.2f})."
        )

    @staticmethod
    def _assess_portfolio_risk(
        weights: Dict[str, float],
        previous: Optional[Dict[str, float]],
        x_state_dict: Dict[str, Any],
    ) -> str:
        """Assess risk of the allocation decision."""
        issues: List[str] = []

        # Concentration risk
        max_share = max(weights.values()) if weights else 0.0
        if max_share > 0.60:
            top_plat = max(weights, key=weights.get)  # type: ignore[arg-type]
            issues.append(
                f"{top_plat} has {max_share:.0%} of budget — high concentration risk"
            )

        # Large shifts
        if previous:
            total_churn = sum(
                abs(weights.get(p, 0.0) - previous.get(p, 0.0))
                for p in set(weights) | set(previous)
            )
            if total_churn > 0.40:
                issues.append(
                    f"total allocation shift of {total_churn:.0%} — "
                    "consider phasing changes over multiple cycles"
                )

        if issues:
            return "WARNING: " + "; ".join(issues) + ". Human review recommended."

        return "Allocation is within normal risk parameters. No human review required."

    # ── reasonability ─────────────────────────────────────────────

    @staticmethod
    def _check_reasonability(action: ActionSpace) -> str:
        issues: List[str] = []
        if abs(action.bid_adjustment) > 0.50:
            issues.append(f"bid change {action.bid_adjustment:+.0%} exceeds +/-50% guardrail")
        if abs(action.budget_adjustment) > 0.30:
            issues.append(f"budget change {action.budget_adjustment:+.0%} exceeds +/-30% guardrail")

        bid_ok = f"A {abs(action.bid_adjustment):.0%} bid change"
        bud_ok = f"a {abs(action.budget_adjustment):.0%} budget change"

        if issues:
            return (
                f"WARNING: {'; '.join(issues)}. "
                "Human review is recommended before applying."
            )

        review = (
            "No human review is required at this confidence level."
            if action.confidence >= 0.70
            else "Human review is recommended due to moderate confidence."
        )

        return (
            f"{bid_ok} and {bud_ok} are both within safe guardrail "
            f"limits (max +/-50% bid, max +/-30% budget). {review}"
        )


# ── Glossary ──────────────────────────────────────────────────────

class ParameterGlossary:
    """Plain-English definitions and expected impact for every key parameter."""

    GLOSSARY: Dict[str, Dict[str, str]] = {
        "roas": {
            "full_name": "Return on Ad Spend",
            "definition": "Revenue generated per dollar spent on advertising.",
            "formula": "total_revenue / total_spend",
            "normal_range": "1.5 - 6.0 (campaign-type dependent)",
            "impact": (
                "Increasing bid or budget when ROAS is high accelerates revenue. "
                "Decreasing budget when ROAS is low reduces waste."
            ),
        },
        "cpa": {
            "full_name": "Cost Per Acquisition",
            "definition": "Average cost to acquire one converting customer.",
            "formula": "total_spend / conversions",
            "normal_range": "$5 - $200 (vertical dependent)",
            "impact": (
                "Lower CPA = more efficient. DRL will reduce bids and refine "
                "audience when CPA rises above target."
            ),
        },
        "ctr": {
            "full_name": "Click-Through Rate",
            "definition": "Percentage of ad impressions that result in a click.",
            "formula": "clicks / impressions",
            "normal_range": "0.5% - 5.0%",
            "impact": (
                "Higher CTR indicates strong ad relevance. Falling CTR triggers "
                "creative rotation or audience refinement."
            ),
        },
        "budget_utilization": {
            "full_name": "Budget Utilisation",
            "definition": "Fraction of the daily or total budget that has been spent.",
            "formula": "total_spend / total_budget",
            "normal_range": "0.0 - 1.0",
            "impact": (
                "High utilisation (>80%) triggers budget scaling decisions. "
                "Low utilisation suggests under-delivery needing bid increases."
            ),
        },
        "creative_fatigue_score": {
            "full_name": "Creative Fatigue Score",
            "definition": "Predicted decay of ad effectiveness as users see it repeatedly.",
            "formula": "ML model output (0-1)",
            "normal_range": "0.0 - 1.0 (>0.65 is elevated)",
            "impact": (
                "Scores above 0.65 predict CTR decline within 48 hours. "
                "DRL triggers creative rotation or testing new variants."
            ),
        },
        "strategic_confidence": {
            "full_name": "Strategic Confidence",
            "definition": "DRL agent's confidence that the recommended action will improve the objective.",
            "formula": "Derived from policy entropy and Q-value spread",
            "normal_range": "0.0 - 1.0 (>0.85 = auto-apply, <0.70 = human review)",
            "impact": (
                "High confidence allows automatic action execution. "
                "Low confidence triggers human review before application."
            ),
        },
        # ── Portfolio / X-Model metrics ──
        "allocation_weight": {
            "full_name": "Platform Allocation Weight",
            "definition": "Fraction of total portfolio budget allocated to a specific platform.",
            "formula": "platform_budget / total_portfolio_budget",
            "normal_range": "0.05 - 0.80 (constrained by min/max share)",
            "impact": (
                "Higher allocation increases a platform's spend and potential reach. "
                "X-Model shifts allocation toward platforms with higher marginal ROAS."
            ),
        },
        "portfolio_hhi": {
            "full_name": "Portfolio Herfindahl-Hirschman Index",
            "definition": "Concentration measure of budget allocation across platforms.",
            "formula": "sum(share_i^2) for each platform i",
            "normal_range": "0.20 - 0.50 (lower = more diversified)",
            "impact": (
                "High HHI indicates over-concentration on few platforms, increasing risk. "
                "Low HHI indicates diversified allocation, reducing single-platform dependency."
            ),
        },
        "marginal_roas": {
            "full_name": "Marginal Return on Ad Spend",
            "definition": "Expected incremental revenue per additional dollar spent on a platform.",
            "formula": "d(revenue) / d(spend) estimated from diminishing-returns curve",
            "normal_range": "0.5 - 5.0 (platform and spend-level dependent)",
            "impact": (
                "X-Model shifts budget toward platforms with higher marginal ROAS, "
                "pulling budget from platforms with diminishing returns."
            ),
        },
    }

    def lookup(self, param_name: str) -> Dict[str, str]:
        """Return the glossary entry for *param_name* (case-insensitive)."""
        key = param_name.strip().lower()
        if key not in self.GLOSSARY:
            return {
                "full_name": param_name,
                "definition": "No glossary entry found.",
                "formula": "N/A",
                "normal_range": "N/A",
                "impact": "N/A",
            }
        return self.GLOSSARY[key]

    def format_entry(self, param_name: str) -> str:
        """Return a human-readable multi-line string for a single entry."""
        e = self.lookup(param_name)
        return (
            f"{e['full_name']} ({param_name})\n"
            f"  Definition  : {e['definition']}\n"
            f"  Formula     : {e['formula']}\n"
            f"  Normal range: {e['normal_range']}\n"
            f"  Impact      : {e['impact']}"
        )
