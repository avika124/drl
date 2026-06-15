"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Row = {
  recommendation: {
    id: string;
    campaign_id: string;
    platform: string;
    status: string;
    confidence_score: number;
    created_at: string;
  };
  points: Array<{ day: number; variance_pct: number; final_impact_score: number }>;
};

export default function HistoryPage() {
  const [rows, setRows] = useState<Row[]>([]);

  useEffect(() => {
    fetch("/api/optimization-history")
      .then((r) => r.json())
      .then((d) => setRows(d.history ?? []))
      .catch(() => setRows([]));
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 p-4">
      <div className="mx-auto max-w-5xl rounded-lg border border-slate-200 bg-white p-4">
        <div className="mb-3 flex items-center justify-between">
          <h1 className="text-lg font-semibold">Optimization History</h1>
          <Link href="/dashboard" className="rounded border px-2 py-1 text-sm">
            Back to Dashboard
          </Link>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-slate-500">
              <tr>
                <th>ID</th>
                <th>Campaign</th>
                <th>Platform</th>
                <th>Status</th>
                <th>Confidence</th>
                <th>Latest Impact</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const latest = row.points.at(-1);
                return (
                  <tr key={row.recommendation.id} className="border-t border-slate-100">
                    <td className="py-2">{row.recommendation.id}</td>
                    <td>{row.recommendation.campaign_id}</td>
                    <td>{row.recommendation.platform}</td>
                    <td>{row.recommendation.status}</td>
                    <td>{(row.recommendation.confidence_score * 100).toFixed(0)}%</td>
                    <td>{latest ? `${latest.final_impact_score}/10` : "Pending"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
