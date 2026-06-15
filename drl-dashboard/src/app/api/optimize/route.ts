import { NextResponse } from "next/server";
import { syncCampaigns, optimizeCampaign } from "@/lib/engine/workflow";
import { ParameterConfig } from "@/lib/types";

export async function POST(req: Request) {
  const body = await req.json();
  const parameters = body.parameters as ParameterConfig | undefined;
  const campaignId = body.campaign_id as string;

  if (!campaignId) {
    return NextResponse.json({ error: "campaign_id is required" }, { status: 400 });
  }

  await syncCampaigns();
  const result = await optimizeCampaign({
    campaign_id: campaignId,
    parameters,
    state: Array.isArray(body.state) ? body.state : undefined,
  });
  if (!result) return NextResponse.json({ error: "Campaign not found" }, { status: 404 });
  return NextResponse.json(result);
}
