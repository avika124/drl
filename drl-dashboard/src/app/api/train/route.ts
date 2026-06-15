import { NextResponse } from "next/server";
import { simulateTrain } from "@/lib/simulations";
import { ParameterConfig } from "@/lib/types";

export async function POST(req: Request) {
  const body = await req.json();
  const parameters = body.parameters as ParameterConfig;
  const result = simulateTrain(parameters);
  return NextResponse.json(result);
}
