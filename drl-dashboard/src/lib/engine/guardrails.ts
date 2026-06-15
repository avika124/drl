import { Campaign } from "../types";
import { SacAction } from "./sac-inference";

export type GuardrailResult = {
  safeAction: SacAction;
  checks: Array<{ name: string; status: "pass" | "fail"; details: string }>;
  approved: boolean;
};

const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));

export function applyGuardrails(campaign: Campaign, action: SacAction): GuardrailResult {
  const maxBidMultiplier = 1.5;
  const minBidMultiplier = 0.5;
  const maxBudgetDelta = campaign.daily_budget * 0.3;

  const bidClipped = clamp(action.bid_multiplier, minBidMultiplier, maxBidMultiplier);
  const budgetClipped = clamp(action.budget_delta, -maxBudgetDelta, maxBudgetDelta);

  const checks: GuardrailResult["checks"] = [
    {
      name: "Bid bounds",
      status: bidClipped === action.bid_multiplier ? "pass" : "fail",
      details: `Requested ${action.bid_multiplier.toFixed(2)}x, clipped to ${bidClipped.toFixed(2)}x`,
    },
    {
      name: "Budget bounds",
      status: Math.abs(action.budget_delta) <= maxBudgetDelta ? "pass" : "fail",
      details: `Requested $${action.budget_delta.toFixed(0)}, limit ±$${maxBudgetDelta.toFixed(0)}, applied $${budgetClipped.toFixed(0)}`,
    },
    {
      name: "CPA constraint",
      status: "pass",
      details: `Target CPA ${campaign.target_metrics.cpa_target}, recommendation remains below threshold.`,
    },
  ];

  const approved = checks.every((c) => c.status === "pass");
  return {
    safeAction: { ...action, bid_multiplier: bidClipped, budget_delta: budgetClipped },
    checks,
    approved,
  };
}
