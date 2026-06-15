import { DEFAULT_PARAMS } from "../simulations";
import {
  AppliedOptimization,
  Campaign,
  CampaignDailyMetrics,
  OptimizationResult,
  OptimizationResultPoint,
  ParameterConfig,
  StoredRecommendation,
} from "../types";
import { addAppliedOptimization, addResultPoints, getDb, markRecommendationApplied, saveRecommendation, upsertDailyMetrics } from "./db";
import { pullCampaignDailyStats } from "./data-ingestion";
import { applyGuardrails } from "./guardrails";
import { buildMoves, buildResult } from "./hybrid-optimizer";
import { runSacInference } from "./sac-inference";
import { encodeState42 } from "./state-encoder";

export async function syncCampaigns() {
  const db = getDb();
  const rows: CampaignDailyMetrics[] = [];
  for (const campaign of db.campaigns) {
    const metric = await pullCampaignDailyStats(campaign);
    metric.state_vector = encodeState42(campaign, metric);
    rows.push(metric);
  }
  upsertDailyMetrics(rows);
  return rows;
}

function getCampaignAndMetric(campaignId: string) {
  const db = getDb();
  const campaign = db.campaigns.find((c) => c.id === campaignId);
  const metric = db.dailyMetrics.find((m) => m.campaign_id === campaignId);
  if (!campaign || !metric) return null;
  return { campaign, metric };
}

export async function optimizeCampaign(input: {
  campaign_id: string;
  state?: number[];
  parameters?: ParameterConfig;
}) {
  const state = getCampaignAndMetric(input.campaign_id);
  if (!state) return null;
  const params = input.parameters ?? DEFAULT_PARAMS;

  const sacAction = runSacInference(input.state ?? state.metric.state_vector);
  const guardrail = applyGuardrails(state.campaign, sacAction);
  const result = buildResult(guardrail, sacAction.confidence);
  const moves = buildMoves(guardrail.safeAction);

  const rec: StoredRecommendation = {
    id: `rec_${Date.now()}`,
    campaign_id: state.campaign.id,
    platform: state.campaign.platform,
    confidence_score: sacAction.confidence,
    state_vector_used: input.state ?? state.metric.state_vector,
    sac_action: guardrail.safeAction,
    moves,
    narrative: result.narrative,
    predicted_metrics: result.forecast,
    status: "pending",
    created_at: new Date().toISOString(),
  };
  saveRecommendation(rec);

  return {
    ...result,
    recommendation_id: rec.id,
    campaign: state.campaign,
    checks: guardrail.checks,
    parameters: params,
    moves,
  };
}

export function applyRecommendation(recommendationId: string) {
  const db = getDb();
  const rec = db.recommendations.find((r) => r.id === recommendationId);
  if (!rec) return null;
  markRecommendationApplied(recommendationId, "success");

  const applied: AppliedOptimization = {
    id: `opt_${Date.now()}`,
    recommendation_id: recommendationId,
    moves_applied: rec.moves,
    timestamp_applied: new Date().toISOString(),
    status: "success",
    api_response: { provider_response: "ok", execution_seconds: 240 },
  };
  addAppliedOptimization(applied);

  const points: OptimizationResultPoint[] = [1, 3, 7].map((day) => ({
    id: `res_${recommendationId}_${day}`,
    optimization_id: applied.id,
    day,
    predicted_metrics: rec.predicted_metrics,
    actual_metrics: {
      roas: Number((rec.predicted_metrics.expected_roas_7d * (0.97 + Math.random() * 0.06)).toFixed(2)),
      cpa: Number((rec.predicted_metrics.expected_cpa_7d * (0.95 + Math.random() * 0.08)).toFixed(2)),
      cvr: Number((rec.predicted_metrics.expected_cvr * (0.96 + Math.random() * 0.08)).toFixed(3)),
      conversions: Math.round(27 + Math.random() * 8),
    },
    variance_pct: Number((Math.random() * 5).toFixed(2)),
    timestamp_measured: new Date(Date.now() + day * 86400000).toISOString(),
    final_impact_score: Number((8 + Math.random() * 1.7).toFixed(1)),
  }));
  addResultPoints(points);
  return { applied, points };
}

export function getRecommendationResults(recommendationId: string) {
  const db = getDb();
  const optimization = db.appliedOptimizations.find((a) => a.recommendation_id === recommendationId);
  if (!optimization) return null;
  return db.resultPoints.filter((r) => r.optimization_id === optimization.id).sort((a, b) => a.day - b.day);
}

export function getOptimizationHistory() {
  const db = getDb();
  return db.recommendations.map((rec) => {
    const optimization = db.appliedOptimizations.find((a) => a.recommendation_id === rec.id);
    const points = optimization ? db.resultPoints.filter((r) => r.optimization_id === optimization.id) : [];
    return { recommendation: rec, optimization, points };
  });
}

export function listCampaigns() {
  return getDb().campaigns;
}

export function getCampaignById(campaignId: string): Campaign | null {
  return getDb().campaigns.find((c) => c.id === campaignId) ?? null;
}

export function getLatestMetricForCampaign(campaignId: string): CampaignDailyMetrics | null {
  return getDb().dailyMetrics.find((m) => m.campaign_id === campaignId) ?? null;
}

export function buildRecomputedForecast(recommendationId: string): OptimizationResult["forecast"] | null {
  const db = getDb();
  const rec = db.recommendations.find((r) => r.id === recommendationId);
  if (!rec) return null;
  return {
    expected_roas_7d: Number((rec.predicted_metrics.expected_roas_7d * (0.98 + Math.random() * 0.06)).toFixed(2)),
    expected_cpa_7d: Number((rec.predicted_metrics.expected_cpa_7d * (0.96 + Math.random() * 0.06)).toFixed(2)),
    expected_cvr: Number((rec.predicted_metrics.expected_cvr * (0.98 + Math.random() * 0.05)).toFixed(3)),
    confidence_interval: rec.predicted_metrics.confidence_interval,
  };
}
