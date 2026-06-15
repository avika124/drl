import { MetricsResponse, OptimizationResult, ParameterConfig, TrainResponse } from "./types";

const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));

export const DEFAULT_PARAMS: ParameterConfig = {
  batch_size: 64,
  gamma: 0.99,
  tau: 0.005,
  max_steps: 1000,
  max_bid_pct: 0.5,
  max_budget_pct: 0.3,
  cooldown: 4,
  reward_roas: 0.3,
  reward_cpa: 0.3,
  reward_conversion: 0.2,
  reward_ctr: 0.2,
  state_dim: 42,
  device: "cpu",
  model_dir: "models/drl",
};

export const PRESETS: Record<"balanced" | "aggressive" | "conservative", ParameterConfig> = {
  balanced: { ...DEFAULT_PARAMS },
  aggressive: {
    ...DEFAULT_PARAMS,
    batch_size: 128,
    max_bid_pct: 0.7,
    max_budget_pct: 0.5,
    reward_roas: 0.5,
    reward_cpa: 0.2,
  },
  conservative: {
    ...DEFAULT_PARAMS,
    batch_size: 32,
    gamma: 0.995,
    max_bid_pct: 0.3,
    max_budget_pct: 0.2,
    reward_roas: 0.35,
    reward_cpa: 0.4,
  },
};

export function simulateTrain(params: ParameterConfig): TrainResponse {
  const lossBase = 0.09 - params.reward_roas * 0.03 + (params.batch_size / 1024) * 0.02;
  return {
    checkpoint: `${params.model_dir}/agent.pt`,
    training_info: {
      final_loss: Number(clamp(lossBase, 0.018, 0.12).toFixed(4)),
      epochs: Math.round(params.max_steps / 100),
      state_dim: params.state_dim,
    },
  };
}

export function simulateOptimize(params: ParameterConfig): OptimizationResult {
  const bidChange = (params.reward_roas * 20 + Math.random() * 4).toFixed(1);
  const budgetIncrease = Math.round(params.max_budget_pct * 1600);
  const confidence = clamp(0.72 + params.reward_roas * 0.25 - params.reward_cpa * 0.08, 0.55, 0.95);
  const roasLift = 2.3 + params.reward_roas * 1.2;
  const cpaDrop = 24.5 - params.reward_cpa * 4.5;

  return {
    directive: {
      bid_change: `+${bidChange}%`,
      budget_change: `$${budgetIncrease}`,
      creative_action: params.reward_ctr > 0.13 ? "rotate" : "hold",
    },
    tactical: {
      headlines: ["Fast Delivery Guaranteed", "Risk-Free 30 Day Trial"],
      descriptions: ["Order today, ship tomorrow", "We stand behind our products"],
    },
    narrative: {
      situation: "Strong CTR, moderate CVR, budget 70% utilized",
      decision: "Increase spend on top-performing audience segments",
      reasoning:
        "When CTR trend and top segment ROAS are both healthy, SAC has learned that controlled bid increases improve reward while keeping CPA within limits.",
      confidence: Number(confidence.toFixed(2)),
      reasonability:
        "Recommendation aligns with ROAS-priority objective and respects guardrail constraints.",
    },
    forecast: {
      expected_roas_7d: Number(roasLift.toFixed(2)),
      expected_cpa_7d: Number(cpaDrop.toFixed(2)),
      expected_cvr: Number((0.032 + params.reward_conversion * 0.03).toFixed(3)),
      confidence_interval: [Number((confidence - 0.05).toFixed(2)), Number((confidence + 0.04).toFixed(2))],
    },
    metrics: [
      { metric: "ROAS", current: "2.3x", predicted: `${roasLift.toFixed(1)}x`, trend: "up" },
      { metric: "CPA", current: "$24.50", predicted: `$${cpaDrop.toFixed(2)}`, trend: "down" },
      { metric: "CTR", current: "2.8%", predicted: "3.1%", trend: "up" },
      { metric: "CVR", current: "3.2%", predicted: `${(3.2 + params.reward_conversion * 1.2).toFixed(1)}%`, trend: "up" },
      { metric: "Volume", current: "1,240", predicted: `${Math.round(1240 + params.max_budget_pct * 850)}`, trend: "up" },
    ],
    execution_time_ms: Math.round(190 + Math.random() * 90),
    schema_version: "1.0",
  };
}

export function simulateMetrics(params: ParameterConfig): MetricsResponse {
  return {
    nodeExecutionTimes: {
      n1: 5,
      n2: 12,
      n3: Math.round(72 + params.batch_size / 4),
      n4: 8,
      n5: 18,
      n6: 9,
      n7: 28,
    },
    parameterImpact: {
      batchVsBuffer: [16, 32, 64, 128, 256, 512].map((x) => ({ x, y: Math.round(x * 1.45 + 40) })),
      gammaVsLoss: [0.9, 0.93, 0.95, 0.97, 0.99, 1.0].map((x) => ({
        x,
        y: Number((0.2 - x * 0.16 + params.tau * 2).toFixed(3)),
      })),
      roasWeightVsTarget: [0, 0.2, 0.4, 0.6, 0.8, 1].map((x) => ({
        x,
        y: Number((1.8 + x * 1.3 - Math.max(0, x - 0.7) * 0.7).toFixed(2)),
      })),
    },
  };
}
