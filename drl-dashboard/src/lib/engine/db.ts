import {
  AppliedOptimization,
  Campaign,
  CampaignDailyMetrics,
  OptimizationResultPoint,
  StoredRecommendation,
} from "../types";

type Database = {
  campaigns: Campaign[];
  dailyMetrics: CampaignDailyMetrics[];
  recommendations: StoredRecommendation[];
  appliedOptimizations: AppliedOptimization[];
  resultPoints: OptimizationResultPoint[];
};

const nowIso = () => new Date().toISOString();

const seedCampaigns: Campaign[] = [
  {
    id: "cmp_7f3a",
    platform: "google_ads",
    name: "Summer Sale - Search",
    status: "active",
    daily_budget: 2000,
    target_metrics: { cpa_target: 25, min_roas: 2.0, daily_budget_limit: 2500 },
  },
  {
    id: "cmp_fb_22",
    platform: "facebook_ads",
    name: "Summer Sale - Collection",
    status: "active",
    daily_budget: 1600,
    target_metrics: { cpa_target: 22, min_roas: 2.1, daily_budget_limit: 2200 },
  },
  {
    id: "cmp_tt_90",
    platform: "tiktok_ads",
    name: "Summer Sale - For You",
    status: "active",
    daily_budget: 1400,
    target_metrics: { cpa_target: 21, min_roas: 2.2, daily_budget_limit: 2000 },
  },
];

const db: Database = {
  campaigns: seedCampaigns,
  dailyMetrics: [],
  recommendations: [],
  appliedOptimizations: [],
  resultPoints: [],
};

export function getDb() {
  return db;
}

export function upsertDailyMetrics(metrics: CampaignDailyMetrics[]) {
  const map = new Map(db.dailyMetrics.map((m) => [m.id, m]));
  metrics.forEach((metric) => map.set(metric.id, metric));
  db.dailyMetrics = Array.from(map.values()).sort((a, b) => b.date.localeCompare(a.date));
  return db.dailyMetrics;
}

export function listLatestCampaignMetrics() {
  return db.campaigns.map((campaign) => {
    const recent = db.dailyMetrics.find((metric) => metric.campaign_id === campaign.id);
    return { campaign, recent };
  });
}

export function saveRecommendation(rec: StoredRecommendation) {
  db.recommendations.unshift(rec);
  const campaign = db.campaigns.find((c) => c.id === rec.campaign_id);
  if (campaign) campaign.last_optimized_at = rec.created_at;
  return rec;
}

export function markRecommendationApplied(recommendationId: string, status: "success" | "failed") {
  const rec = db.recommendations.find((r) => r.id === recommendationId);
  if (!rec) return null;
  rec.status = status === "success" ? "applied" : "archived";
  rec.applied_at = nowIso();
  return rec;
}

export function addAppliedOptimization(row: AppliedOptimization) {
  db.appliedOptimizations.unshift(row);
  return row;
}

export function addResultPoints(points: OptimizationResultPoint[]) {
  db.resultPoints.unshift(...points);
  return points;
}
