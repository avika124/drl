import { NextResponse } from "next/server";
import { buildRecomputedForecast } from "@/lib/engine/workflow";

export async function POST(req: Request) {
  const body = await req.json();
  const recommendationId = body.recommendation_id as string;
  if (!recommendationId) return NextResponse.json({ error: "recommendation_id is required" }, { status: 400 });
  const forecast = buildRecomputedForecast(recommendationId);
  if (!forecast) return NextResponse.json({ error: "Recommendation not found" }, { status: 404 });
  return NextResponse.json({ recommendation_id: recommendationId, forecast });
}
