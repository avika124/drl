"use client";

import { useEffect, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useWorkflow } from "@/hooks/useWorkflow";
import { MetricsResponse } from "@/lib/types";

export function ResultsDisplay() {
  const { lastResult } = useWorkflow();
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  useEffect(() => {
    fetch("/api/metrics")
      .then((res) => res.json())
      .then(setMetrics)
      .catch(() => undefined);
  }, []);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3">
      <h2 className="mb-2 text-sm font-semibold">Optimization Results</h2>
      {!lastResult ? (
        <p className="text-xs text-slate-500">Run workflow to see directives, forecast, narrative, and metrics.</p>
      ) : (
        <div className="space-y-3 text-sm">
          <div className="rounded border border-slate-200 p-2">
            <p className="font-medium">Directive</p>
            <ul className="mt-1 list-disc pl-5 text-xs text-slate-700">
              <li>Increase bid: {lastResult.directive.bid_change}</li>
              <li>Increase budget: {lastResult.directive.budget_change}</li>
              <li>Creative action: {lastResult.directive.creative_action}</li>
            </ul>
          </div>
          <div className="rounded border border-slate-200 p-2 text-xs text-slate-700">
            <p className="font-medium">Narrative</p>
            <p className="mt-1">{lastResult.narrative.situation}</p>
            <p>{lastResult.narrative.decision}</p>
            <p>{lastResult.narrative.reasoning}</p>
          </div>
          <div className="rounded border border-slate-200 p-2">
            <p className="mb-1 text-xs font-medium">Performance metrics</p>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-slate-500">
                    <th>Metric</th>
                    <th>Current</th>
                    <th>Predicted</th>
                  </tr>
                </thead>
                <tbody>
                  {lastResult.metrics.map((metric) => (
                    <tr key={metric.metric}>
                      <td>{metric.metric}</td>
                      <td>{metric.current}</td>
                      <td>{metric.predicted}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {metrics && (
        <div className="mt-3 grid gap-3 md:grid-cols-3">
          <ImpactGraph title="Batch Size → Buffer" data={metrics.parameterImpact.batchVsBuffer} xKey="x" yKey="y" />
          <ImpactGraph title="Gamma → Policy Loss" data={metrics.parameterImpact.gammaVsLoss} xKey="x" yKey="y" />
          <ImpactGraph title="ROAS Weight → Target" data={metrics.parameterImpact.roasWeightVsTarget} xKey="x" yKey="y" />
        </div>
      )}
    </section>
  );
}

function ImpactGraph({
  title,
  data,
  xKey,
  yKey,
}: {
  title: string;
  data: Array<{ x: number; y: number }>;
  xKey: "x";
  yKey: "y";
}) {
  return (
    <div className="h-44 rounded border border-slate-200 p-2">
      <p className="mb-1 text-xs font-medium">{title}</p>
      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={data}>
          <XAxis dataKey={xKey} fontSize={11} />
          <YAxis fontSize={11} />
          <Tooltip />
          <Line type="monotone" dataKey={yKey} stroke="#2563eb" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
