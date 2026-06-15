import { NextResponse } from "next/server";
import { getLatestMetricForCampaign, listCampaigns, syncCampaigns } from "@/lib/engine/workflow";

export async function GET() {
  await syncCampaigns();
  const campaigns = listCampaigns();
  return NextResponse.json({
    campaigns: campaigns.map((campaign) => ({
      ...campaign,
      latest: getLatestMetricForCampaign(campaign.id),
    })),
  });
}
