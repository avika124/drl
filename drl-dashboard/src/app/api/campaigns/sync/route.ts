import { NextResponse } from "next/server";
import { listCampaigns, syncCampaigns } from "@/lib/engine/workflow";

export async function POST() {
  const metrics = await syncCampaigns();
  const campaigns = listCampaigns();
  return NextResponse.json({
    campaigns,
    metrics_synced: metrics.length,
    latest: metrics,
  });
}
