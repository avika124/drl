import { NextResponse } from "next/server";
import { getRecommendationResults } from "@/lib/engine/workflow";

export async function GET(_req: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const points = getRecommendationResults(id);
  if (!points) return NextResponse.json({ error: "Results not found" }, { status: 404 });
  return NextResponse.json({ recommendation_id: id, points });
}
