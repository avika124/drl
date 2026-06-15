"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Campaign, StoredRecommendation } from "@/lib/types";

type CampaignWithLatest = Campaign & {
  latest?: {
    roas: number;
    cpa: number;
    spend: number;
    conversion_value: number;
  } | null;
};

type HistoryItem = {
  recommendation: StoredRecommendation;
  points: Array<{ day: number; variance_pct: number; final_impact_score: number }>;
};

export default function DashboardPage() {
  const [campaigns, setCampaigns] = useState<CampaignWithLatest[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [activeRecommendationId, setActiveRecommendationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    const [campaignRes, historyRes] = await Promise.all([fetch("/api/campaigns"), fetch("/api/optimization-history")]);
    const campaignJson = await campaignRes.json();
    const historyJson = await historyRes.json();
    setCampaigns(campaignJson.campaigns ?? []);
    setHistory(historyJson.history ?? []);
  };

  useEffect(() => {
    refresh();
  }, []);

  const totals = useMemo(() => {
    const spend = campaigns.reduce((sum, c) => sum + (c.latest?.spend ?? 0), 0);
    const revenue = campaigns.reduce((sum, c) => sum + (c.latest?.conversion_value ?? 0), 0);
    const roas = spend > 0 ? revenue / spend : 0;
    const cpa =
      campaigns.reduce((sum, c) => sum + (c.latest?.cpa ?? 0), 0) / Math.max(campaigns.filter((c) => c.latest).length, 1);
    return { spend, revenue, roas, cpa };
  }, [campaigns]);

  const requestRecommendations = async () => {
    setLoading(true);
    try {
      await fetch("/api/campaigns/sync", { method: "POST" });
      if (!campaigns.length) return;
      const target = campaigns[0];
      const res = await fetch("/api/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ campaign_id: target.id }),
      });
      const data = await res.json();
      setActiveRecommendationId(data.recommendation_id ?? null);
      await refresh();
    } finally {
      setLoading(false);
    }
  };

  const applyRecommendation = async () => {
    if (!activeRecommendationId) return;
    await fetch("/api/recommendations/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recommendation_id: activeRecommendationId }),
    });
    await refresh();
  };

  return (
    <div className="min-h-screen bg-slate-50 p-4 text-slate-900">
      <div className="mx-auto max-w-[1400px]">
        <header className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white p-3">
          <div>
            <h1 className="text-lg font-semibold">DRL Campaign Optimizer</h1>
            <p className="text-xs text-slate-500">Real data sync → DRL recommendation → apply → track outcomes</p>
          </div>
          <div className="flex gap-2 text-sm">
            <Link className="rounded border px-2 py-1" href="/history">
              View History
            </Link>
            <Link className="rounded border px-2 py-1" href="/settings">
              Settings
            </Link>
            <button onClick={requestRecommendations} className="rounded bg-blue-600 px-3 py-1 text-white disabled:opacity-70" disabled={loading}>
              {loading ? "Running..." : "Get Recommendations"}
            </button>
          </div>
        </header>

        <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
          <section className="space-y-4">
            <div className="rounded-lg border border-slate-200 bg-white p-3">
              <h2 className="text-sm font-semibold">Quick Stats</h2>
              <div className="mt-2 space-y-1 text-sm">
                <p>Total ROAS: {totals.roas.toFixed(2)}x</p>
                <p>Total CPA: ${totals.cpa.toFixed(2)}</p>
                <p>Daily Spend: ${totals.spend.toFixed(0)}</p>
                <p>Daily Revenue: ${totals.revenue.toFixed(0)}</p>
                <p>Daily Profit: ${(totals.revenue - totals.spend).toFixed(0)}</p>
                <p>Optimizations Applied: {history.filter((h) => h.recommendation.status === "applied").length}</p>
              </div>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-3">
              <h2 className="text-sm font-semibold">Recent Optimizations</h2>
              <div className="mt-2 space-y-2 text-xs">
                {history.slice(0, 4).map((item) => (
                  <div key={item.recommendation.id} className="rounded border border-slate-200 p-2">
                    <p className="font-medium">{item.recommendation.campaign_id}</p>
                    <p>Status: {item.recommendation.status}</p>
                    <p>Confidence: {(item.recommendation.confidence_score * 100).toFixed(0)}%</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="space-y-4">
            <div className="rounded-lg border border-slate-200 bg-white p-3">
              <h2 className="text-sm font-semibold">Active Campaigns</h2>
              <div className="mt-2 grid gap-2 md:grid-cols-2">
                {campaigns.map((campaign) => (
                  <Link key={campaign.id} href={`/campaign/${campaign.id}`} className="rounded border border-slate-200 p-2 text-sm hover:bg-slate-50">
                    <p className="font-medium">{campaign.name}</p>
                    <p className="text-xs text-slate-600">{campaign.platform}</p>
                    <p className="text-xs">ROAS: {campaign.latest?.roas ?? "—"}</p>
                    <p className="text-xs">CPA: ${campaign.latest?.cpa ?? "—"}</p>
                  </Link>
                ))}
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-white p-3">
              <h2 className="text-sm font-semibold">Current Optimization</h2>
              {activeRecommendationId ? (
                <div className="mt-2 text-sm">
                  <p>Recommendation ID: {activeRecommendationId}</p>
                  <p className="text-xs text-slate-600">Ready to apply. System will track day 1/3/7 variance automatically.</p>
                  <button onClick={applyRecommendation} className="mt-2 rounded bg-emerald-600 px-3 py-1 text-white">
                    Apply Recommendation
                  </button>
                </div>
              ) : (
                <p className="mt-2 text-sm text-slate-600">No active recommendation. Click &quot;Get Recommendations&quot;.</p>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
