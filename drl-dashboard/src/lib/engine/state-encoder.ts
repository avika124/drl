import { Campaign, CampaignDailyMetrics } from "../types";

const norm = (value: number, max = 1) => Math.max(0, Math.min(1, value / Math.max(max, 1)));

export function encodeState42(campaign: Campaign, metric: CampaignDailyMetrics): number[] {
  const hour = new Date().getHours() / 23;
  const day = new Date().getDay() / 6;
  const dayOfMonth = (new Date().getDate() - 1) / 30;
  const isWeekend = new Date().getDay() === 0 || new Date().getDay() === 6 ? 1 : 0;

  const vec = [
    metric.ctr,
    metric.cvr,
    norm(metric.roas, 6),
    norm(metric.cpa, 120),
    norm(metric.cpc, 25),
    norm(metric.cpm, 80),
    norm(metric.spend / campaign.daily_budget, 2),
    norm(metric.impressions, 600000),
    norm(metric.clicks, 20000),
    norm(metric.conversions, 3000),
    hour,
    day,
    dayOfMonth,
    isWeekend,
    0,
    0.4,
    0.53,
    0.49,
    0.58,
    0.46,
    0.52,
    0.33,
    0.44,
    0.41,
    0.62,
    0.31,
    0.55,
    0.49,
    0.71,
    campaign.target_metrics.min_roas > 2 ? 0.5 : 0.25,
    campaign.platform === "google_ads" ? 0.2 : campaign.platform === "facebook_ads" ? 0.4 : 0.6,
    0.7,
    norm(metric.spend / campaign.daily_budget, 1.2),
    norm(Math.log1p(metric.spend), 10),
    norm(Math.log1p(metric.spend * 30), 14),
    norm(Math.log1p(campaign.daily_budget), 10),
    0.4,
    norm(metric.roas, 6),
    0.22,
    norm(campaign.target_metrics.cpa_target, 80),
    norm(campaign.target_metrics.min_roas, 6),
    norm(campaign.target_metrics.daily_budget_limit, 10000),
  ];
  return vec.map((value) => Number(value.toFixed(4)));
}
