"use client";

import { PlayCircle } from "lucide-react";
import { useWorkflow } from "@/hooks/useWorkflow";

export function Header() {
  const { isRunning, runAll } = useWorkflow();

  return (
    <header className="border-b border-slate-200 bg-slate-900 px-4 py-3 text-white">
      <div className="mx-auto flex max-w-[1500px] items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold">DRL Cross-Platform Optimizer</h1>
          <p className="text-xs text-slate-300">M1 Training → M2 Inference visual execution dashboard</p>
        </div>
        <button
          onClick={runAll}
          disabled={isRunning}
          className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-3 py-2 text-sm font-medium hover:bg-blue-500 disabled:opacity-60"
        >
          <PlayCircle className="h-4 w-4" />
          {isRunning ? "Running..." : "Run All"}
        </button>
      </div>
    </header>
  );
}
