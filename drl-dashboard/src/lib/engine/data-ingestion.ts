import { Campaign, CampaignDailyMetrics, Platform } from "../types";

type RawRow = Omit<CampaignDailyMetrics, "id" | "campaign_id" | "state_vector">;

const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));

function makeRaw(baseSpend: number, baseConv: number): RawRow {
  const spend = baseSpend + Math.random() * 250;
  const impressions = Math.round(spend * (62 + Math.random() * 18));
  const clicks = Math.round(impressions * (0.024 + Math.random() * 0.01));
  const conversions = Math.round(baseConv + Math.random() * 40);
  const conversionValue = conversions * (130 + Math.random() * 35);
  const ctr = clicks / Math.max(impressions, 1);
  const cvr = conversions / Math.max(clicks, 1);
  const cpa = spend / Math.max(conversions, 1);
  const roas = conversionValue / Math.max(spend, 1);
  const cpm = (spend * 1000) / Math.max(impressions, 1);
  const cpc = spend / Math.max(clicks, 1);

  return {
    date: new Date().toISOString().slice(0, 10),
    spend: Number(spend.toFixed(2)),
    impressions,
    clicks,
    conversions,
    conversion_value: Number(conversionValue.toFixed(2)),
    ctr: Number(clamp(ctr, 0, 1).toFixed(4)),
    cvr: Number(clamp(cvr, 0, 1).toFixed(4)),
    cpa: Number(cpa.toFixed(2)),
    roas: Number(roas.toFixed(2)),
    cpm: Number(cpm.toFixed(2)),
    cpc: Number(cpc.toFixed(2)),
  };
}

export async function pullCampaignDailyStats(campaign: Campaign): Promise<CampaignDailyMetrics> {
  const seedByPlatform: Record<Platform, { spend: number; conv: number }> = {
    google_ads: { spend: 5400, conv: 240 },
    facebook_ads: { spend: 3800, conv: 210 },
    tiktok_ads: { spend: 3200, conv: 185 },
    linkedin_ads: { spend: 1700, conv: 70 },
  };
  const seed = seedByPlatform[campaign.platform];
  const row = makeRaw(seed.spend, seed.conv);
  return {
    id: `met_${campaign.id}_${Date.now()}`,
    campaign_id: campaign.id,
    state_vector: [],
    ...row,
  };
}
