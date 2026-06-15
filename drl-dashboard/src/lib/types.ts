export type PresetName = "balanced" | "aggressive" | "conservative";

export type NodeStatus = "pending" | "running" | "success" | "error";

export type WorkflowNode =
  | "n1"
  | "n2"
  | "n3"
  | "n4"
  | "n5"
  | "n6"
  | "n7";

export interface ParameterConfig {
  batch_size: number;
  gamma: number;
  tau: number;
  max_steps: number;
  max_bid_pct: number;
  max_budget_pct: number;
  cooldown: number;
  reward_roas: number;
  reward_cpa: number;
  reward_conversion: number;
  reward_ctr: number;
  state_dim: number;
  device: "cpu" | "gpu";
  model_dir: string;
}

export interface LogEntry {
  id: string;
  at: string;
  level: "info" | "warn" | "error";
  message: string;
}

export interface OptimizationResult {
  directive: {
    bid_change: string;
    budget_change: string;
    creative_action: "hold" | "rotate" | "test_new";
  };
  tactical: {
    headlines: string[];
    descriptions: string[];
  };
  narrative: {
    situation: string;
    decision: string;
    reasoning: string;
    confidence: number;
    reasonability: string;
  };
  forecast: {
    expected_roas_7d: number;
    expected_cpa_7d: number;
    expected_cvr: number;
    confidence_interval: [number, number];
  };
  metrics: Array<{
    metric: string;
    current: string;
    predicted: string;
    trend: "up" | "down" | "flat";
  }>;
  execution_time_ms: number;
  schema_version: string;
}

export interface TrainResponse {
  checkpoint: string;
  training_info: {
    final_loss: number;
    epochs: number;
    state_dim: number;
  };
}

export interface MetricsResponse {
  nodeExecutionTimes: Record<WorkflowNode, number>;
  parameterImpact: {
    batchVsBuffer: Array<{ x: number; y: number }>;
    gammaVsLoss: Array<{ x: number; y: number }>;
    roasWeightVsTarget: Array<{ x: number; y: number }>;
  };
}

export type Platform = "google_ads" | "facebook_ads" | "tiktok_ads" | "linkedin_ads";

export interface Campaign {
  id: string;
  platform: Platform;
  name: string;
  status: "active" | "paused";
  daily_budget: number;
  target_metrics: { cpa_target: number; min_roas: number; daily_budget_limit: number };
  last_optimized_at?: string;
}

export interface CampaignDailyMetrics {
  id: string;
  campaign_id: string;
  date: string;
  spend: number;
  impressions: number;
  clicks: number;
  conversions: number;
  conversion_value: number;
  ctr: number;
  cvr: number;
  cpa: number;
  roas: number;
  cpm: number;
  cpc: number;
  state_vector: number[];
}

export interface RecommendationMove {
  type: "bid_increase" | "budget_increase" | "creative_swap" | "audience_expand";
  params: Record<string, string | number | boolean>;
  reasoning: string;
}

export interface StoredRecommendation {
  id: string;
  campaign_id: string;
  platform: Platform;
  confidence_score: number;
  state_vector_used: number[];
  sac_action: {
    bid_multiplier: number;
    budget_delta: number;
    creative_id: number;
    audience_action: "hold" | "expand" | "refine";
  };
  moves: RecommendationMove[];
  narrative: OptimizationResult["narrative"];
  predicted_metrics: OptimizationResult["forecast"];
  status: "pending" | "applied" | "archived";
  created_at: string;
  applied_at?: string;
}

export interface AppliedOptimization {
  id: string;
  recommendation_id: string;
  moves_applied: RecommendationMove[];
  timestamp_applied: string;
  status: "success" | "failed";
  api_response: Record<string, unknown>;
}

export interface OptimizationResultPoint {
  id: string;
  optimization_id: string;
  day: number;
  predicted_metrics: OptimizationResult["forecast"];
  actual_metrics: {
    roas: number;
    cpa: number;
    cvr: number;
    conversions: number;
  };
  variance_pct: number;
  timestamp_measured: string;
  final_impact_score: number;
}
