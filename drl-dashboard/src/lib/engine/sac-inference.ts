export type SacAction = {
  bid_multiplier: number;
  budget_delta: number;
  creative_id: number;
  audience_action: "hold" | "expand" | "refine";
  q_value: number;
  confidence: number;
};

const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));

export function runSacInference(stateVector: number[]): SacAction {
  const ctr = stateVector[0];
  const cvr = stateVector[1];
  const roas = stateVector[2] * 6;
  const fatigue = stateVector[25];
  const topSegmentRoas = stateVector[37] * 6;
  const budgetUtil = stateVector[32];

  const bidLift = clamp(1 + (ctr + cvr + topSegmentRoas / 6) * 0.25, 0.8, 1.5);
  const budgetDelta = clamp((roas > 2 ? 1 : -1) * (180 + budgetUtil * 620), -600, 700);
  const creative = fatigue > 0.45 ? 3 : 1;
  const audienceAction: SacAction["audience_action"] = topSegmentRoas > 3 ? "expand" : "hold";
  const qValue = clamp(0.9 + roas * 0.55 + ctr * 0.6 + cvr * 0.45, -2, 4);
  const confidence = 1 / (1 + Math.exp(-qValue / 2.2));

  return {
    bid_multiplier: Number(bidLift.toFixed(3)),
    budget_delta: Number(budgetDelta.toFixed(2)),
    creative_id: creative,
    audience_action: audienceAction,
    q_value: Number(qValue.toFixed(3)),
    confidence: Number(confidence.toFixed(3)),
  };
}
