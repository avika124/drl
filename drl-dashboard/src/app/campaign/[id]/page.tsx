"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

type CampaignData = {
  id: string;
  name: string;
  platform: string;
  status: string;
  daily_budget: number;
  latest?: {
    roas: number;
    cpa: number;
    ctr: number;
    cvr: number;
    spend: number;
    conversions: number;
  };
};

export default function CampaignDetailPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [campaign, setCampaign] = useState<CampaignData | null>(null);

  useEffect(() => {
    fetch("/api/campaigns")
      .then((r) => r.json())
      .then((d) => {
        const match = (d.campaigns ?? []).find((item: CampaignData) => item.id === id);
        setCampaign(match ?? null);
      });
  }, [id]);

  return (
    <div className="min-h-screen bg-slate-50 p-4">
      <div className="mx-auto max-w-4xl rounded-lg border border-slate-200 bg-white p-4">
        <div className="mb-3 flex items-center justify-between">
          <h1 className="text-lg font-semibold">Campaign Detail</h1>
          <Link href="/dashboard" className="rounded border px-2 py-1 text-sm">
            Back
          </Link>
        </div>
        {!campaign ? (
          <p className="text-sm text-slate-600">Campaign not found.</p>
        ) : (
          <div className="space-y-2 text-sm">
            <p className="font-medium">{campaign.name}</p>
            <p>Platform: {campaign.platform}</p>
            <p>Status: {campaign.status}</p>
            <p>Daily budget: ${campaign.daily_budget}</p>
            <div className="mt-3 grid gap-2 md:grid-cols-3">
              <MetricCard label="ROAS" value={`${campaign.latest?.roas ?? "—"}x`} />
              <MetricCard label="CPA" value={`$${campaign.latest?.cpa ?? "—"}`} />
              <MetricCard label="CTR" value={`${campaign.latest?.ctr ?? "—"}%`} />
              <MetricCard label="CVR" value={`${campaign.latest?.cvr ?? "—"}%`} />
              <MetricCard label="Spend" value={`$${campaign.latest?.spend ?? "—"}`} />
              <MetricCard label="Conversions" value={`${campaign.latest?.conversions ?? "—"}`} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-slate-200 p-2">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-sm font-medium">{value}</p>
    </div>
  );
}
