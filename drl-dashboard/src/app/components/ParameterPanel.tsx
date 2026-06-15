"use client";

import { ChangeEvent, useRef } from "react";
import { Download, Upload } from "lucide-react";
import { ParameterConfig, PresetName } from "@/lib/types";
import { useWorkflow } from "@/hooks/useWorkflow";

const groups: Array<{
  label: string;
  fields: Array<{ key: keyof ParameterConfig; min?: number; max?: number; step?: number; fixed?: boolean }>;
}> = [
  {
    label: "Training",
    fields: [
      { key: "batch_size", min: 16, max: 512, step: 16 },
      { key: "gamma", min: 0.9, max: 1, step: 0.001 },
      { key: "tau", min: 0.001, max: 0.02, step: 0.001 },
      { key: "max_steps", min: 200, max: 5000, step: 100 },
    ],
  },
  {
    label: "Safety",
    fields: [
      { key: "max_bid_pct", min: 0.1, max: 1, step: 0.05 },
      { key: "max_budget_pct", min: 0.1, max: 1, step: 0.05 },
      { key: "cooldown", min: 1, max: 24, step: 1 },
    ],
  },
  {
    label: "Rewards",
    fields: [
      { key: "reward_roas", min: 0, max: 1, step: 0.05 },
      { key: "reward_cpa", min: 0, max: 1, step: 0.05 },
      { key: "reward_conversion", min: 0, max: 1, step: 0.05 },
      { key: "reward_ctr", min: 0, max: 1, step: 0.05 },
    ],
  },
  {
    label: "Data",
    fields: [
      { key: "state_dim", fixed: true },
      { key: "model_dir" },
    ],
  },
];

const human = (v: string) => v.replaceAll("_", " ");

export function ParameterPanel() {
  const { parameters, baselineParameters, modifiedCount, setParameter, loadPreset, exportConfig, importConfigFile } = useWorkflow();
  const fileRef = useRef<HTMLInputElement>(null);

  const onImport = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) await importConfigFile(file);
    if (fileRef.current) fileRef.current.value = "";
  };

  const presetBtn = (name: PresetName) => (
    <button
      key={name}
      onClick={() => loadPreset(name)}
      className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs font-medium capitalize hover:bg-slate-50"
    >
      {name}
    </button>
  );

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold">Parameters</h2>
        <span className="rounded bg-amber-100 px-2 py-0.5 text-xs text-amber-700">Modified: {modifiedCount}</span>
      </div>
      <div className="mb-3 flex gap-2">{(["balanced", "aggressive", "conservative"] as PresetName[]).map(presetBtn)}</div>

      <div className="space-y-2">
        {groups.map((group) => (
          <details key={group.label} open className="rounded border border-slate-200">
            <summary className="cursor-pointer px-2 py-1 text-sm font-medium">{group.label}</summary>
            <div className="space-y-2 px-2 pb-2">
              {group.fields.map((field) => {
                const key = field.key;
                const value = parameters[key];
                const changed = value !== baselineParameters[key];
                const isNumber = typeof value === "number";
                return (
                  <label key={key} className={`block rounded p-1 ${changed ? "bg-amber-50" : ""}`}>
                    <span className="mb-1 block text-xs capitalize text-slate-600">{human(key)}</span>
                    {field.fixed ? (
                      <input value={String(value)} disabled className="w-full rounded border border-slate-200 bg-slate-100 px-2 py-1 text-sm" />
                    ) : key === "device" ? (
                      <select
                        value={String(value)}
                        onChange={(e) => setParameter(key, e.target.value as ParameterConfig[typeof key])}
                        className="w-full rounded border border-slate-300 px-2 py-1 text-sm"
                      >
                        <option value="cpu">CPU</option>
                        <option value="gpu">GPU</option>
                      </select>
                    ) : (
                      <input
                        type={isNumber ? "number" : "text"}
                        min={field.min}
                        max={field.max}
                        step={field.step}
                        value={String(value)}
                        onChange={(e) => {
                          const next = isNumber ? Number(e.target.value) : e.target.value;
                          setParameter(key, next as ParameterConfig[typeof key]);
                        }}
                        className="w-full rounded border border-slate-300 px-2 py-1 text-sm"
                      />
                    )}
                  </label>
                );
              })}
            </div>
          </details>
        ))}
      </div>

      <div className="mt-3 flex gap-2">
        <button onClick={exportConfig} className="inline-flex items-center gap-1 rounded border px-2 py-1 text-xs">
          <Download className="h-3.5 w-3.5" /> Export
        </button>
        <button onClick={() => fileRef.current?.click()} className="inline-flex items-center gap-1 rounded border px-2 py-1 text-xs">
          <Upload className="h-3.5 w-3.5" /> Import
        </button>
        <input ref={fileRef} type="file" hidden accept="application/json" onChange={onImport} />
      </div>
    </section>
  );
}
