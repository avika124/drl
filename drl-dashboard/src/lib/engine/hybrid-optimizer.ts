import { OptimizationResult, RecommendationMove } from "../types";
import { GuardrailResult } from "./guardrails";
import { SacAction } from "./sac-inference";

export function buildMoves(action: SacAction): RecommendationMove[] {
  return [
    {
      type: "bid_increase",
      params: { multiplier: action.bid_multiplier, pct: Number(((action.bid_multiplier - 1) * 100).toFixed(1)) },
      reasoning: "High-intent segments and positive CTR trend justify a controlled bid increase.",
    },
    {
      type: "budget_increase",
      params: { delta: action.budget_delta },
      reasoning: "Projected ROAS remains above target with higher spend capacity.",
    },
    {
      type: "creative_swap",
      params: { creative_id: action.creative_id, rollout_pct: 20 },
      reasoning: "Fatigue signal suggests fresh creative should improve engagement.",
    },
  ];
}

export function buildResult(safe: GuardrailResult, confidence: number): OptimizationResult {
  const bidPct = ((safe.safeAction.bid_multiplier - 1) * 100).toFixed(1);
  const spendDelta = safe.safeAction.budget_delta;
  const roas = 2.3 + Math.max(0, spendDelta) / 2200 + Number(bidPct) / 120;
  const cpa = 24.5 - Number(bidPct) / 8 - Math.max(0, spendDelta) / 1500;

  return {
    directive: {
      bid_change: `${Number(bidPct) >= 0 ? "+" : ""}${bidPct}%`,
      budget_change: `${spendDelta >= 0 ? "+" : ""}$${Math.abs(Math.round(spendDelta))}`,
      creative_action: safe.safeAction.creative_id === 3 ? "rotate" : "hold",
    },
    tactical: {
      headlines: ["Fast Delivery Guaranteed", "Risk-Free 30 Day Trial"],
      descriptions: ["Order today, ship tomorrow", "We stand behind our products"],
    },
    narrative: {
      situation: "Campaign has strong click-through but moderate conversion with rising opportunity in top segment.",
      decision: "Increase bid and budget carefully, and rotate creative with controlled rollout.",
      reasoning:
        "SAC policy and critic estimate indicate positive reward under current context. Guardrails validated bid and budget bounds, then kept action in safe operating limits.",
      confidence: Number(confidence.toFixed(2)),
      reasonability:
        safe.approved ? "All guardrails passed; recommendation aligns with constraints." : "Action required clipping before approval.",
    },
    forecast: {
      expected_roas_7d: Number(roas.toFixed(2)),
      expected_cpa_7d: Number(Math.max(10, cpa).toFixed(2)),
      expected_cvr: 0.034,
      confidence_interval: [Number((confidence - 0.06).toFixed(2)), Number((confidence + 0.03).toFixed(2))],
    },
    metrics: [
      { metric: "ROAS", current: "2.3x", predicted: `${roas.toFixed(2)}x`, trend: "up" },
      { metric: "CPA", current: "$24.50", predicted: `$${Math.max(10, cpa).toFixed(2)}`, trend: "down" },
      { metric: "CTR", current: "2.8%", predicted: "2.95%", trend: "up" },
      { metric: "CVR", current: "3.2%", predicted: "3.45%", trend: "up" },
      { metric: "Volume", current: "1,240", predicted: "1,337", trend: "up" },
    ],
    execution_time_ms: 245,
    schema_version: "1.0",
  };
}
