import Link from "next/link";

export default function SettingsPage() {
  return (
    <div className="min-h-screen bg-slate-50 p-4">
      <div className="mx-auto max-w-3xl rounded-lg border border-slate-200 bg-white p-4">
        <div className="mb-3 flex items-center justify-between">
          <h1 className="text-lg font-semibold">Integration Settings</h1>
          <Link href="/dashboard" className="rounded border px-2 py-1 text-sm">
            Back
          </Link>
        </div>
        <p className="mb-3 text-sm text-slate-600">
          Configure API keys via environment variables for production connectors. This MVP currently runs simulated connectors with real pipeline
          orchestration.
        </p>
        <div className="space-y-2 text-sm">
          <SettingRow name="Google Ads" envKey="GOOGLE_ADS_API_KEY" />
          <SettingRow name="Facebook Ads" envKey="FACEBOOK_ADS_API_KEY" />
          <SettingRow name="TikTok Ads" envKey="TIKTOK_ADS_API_KEY" />
          <SettingRow name="LinkedIn Ads" envKey="LINKEDIN_ADS_API_KEY" />
          <SettingRow name="Database URL" envKey="DATABASE_URL" />
        </div>
      </div>
    </div>
  );
}

function SettingRow({ name, envKey }: { name: string; envKey: string }) {
  return (
    <div className="rounded border border-slate-200 p-2">
      <p className="font-medium">{name}</p>
      <p className="text-xs text-slate-500">Env: {envKey}</p>
    </div>
  );
}
