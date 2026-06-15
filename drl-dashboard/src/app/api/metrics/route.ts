import { NextResponse } from "next/server";
import { DEFAULT_PARAMS, simulateMetrics } from "@/lib/simulations";

export async function GET() {
  return NextResponse.json(simulateMetrics(DEFAULT_PARAMS));
}
