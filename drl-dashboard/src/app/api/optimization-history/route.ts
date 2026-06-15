import { NextResponse } from "next/server";
import { getOptimizationHistory } from "@/lib/engine/workflow";

export async function GET() {
  return NextResponse.json({ history: getOptimizationHistory() });
}
