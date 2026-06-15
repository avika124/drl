"use client";

import { Download, RefreshCw } from "lucide-react";
import { useWorkflow } from "@/hooks/useWorkflow";
import { WorkflowNode } from "@/lib/types";

const nodeLabel: Record<WorkflowNode, string> = {
  n1: "campaign env",
  n2: "replay buffer",
  n3: "SAC training",
  n4: "checkpoint",
  n5: "load SAC",
  n6: "safety guardrails",
  n7: "hybrid optimizer",
};

export function ExecutionMonitor() {
  const { isRunning, progress, nodeStatuses, nodeTimes, executionLogs, resetExecution } = useWorkflow();

  const downloadLogs = () => {
    const content = executionLogs.map((l) => `[${l.at}] ${l.message}`).join("\n");
    const blob = new Blob([content], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "execution.log";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold">Execution Monitor</h2>
        <span className={`rounded px-2 py-0.5 text-xs ${isRunning ? "bg-amber-100 text-amber-700" : "bg-emerald-100 text-emerald-700"}`}>
          {isRunning ? "Running" : "Idle"}
        </span>
      </div>
      <div className="mb-2 h-2 overflow-hidden rounded bg-slate-100">
        <div className="h-full bg-blue-600 transition-all" style={{ width: `${progress}%` }} />
      </div>
      <p className="mb-2 text-xs text-slate-500">Progress: {progress}%</p>

      <div className="space-y-1 text-xs">
        {(Object.keys(nodeLabel) as WorkflowNode[]).map((node) => (
          <div key={node} className="flex items-center justify-between rounded border border-slate-200 px-2 py-1">
            <span>
              {node} ({nodeLabel[node]})
            </span>
            <span>{nodeStatuses[node] === "success" ? `${nodeTimes[node] ?? 0}ms` : nodeStatuses[node]}</span>
          </div>
        ))}
      </div>

      <div className="mt-3 rounded border border-slate-200 bg-slate-50 p-2">
        <p className="mb-1 text-xs font-medium">Live logs</p>
        <div className="h-28 overflow-auto rounded bg-slate-900 p-2 font-mono text-[11px] text-slate-100">
          {executionLogs.map((log) => (
            <div key={log.id}>
              [{log.at}] {log.message}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-2 flex gap-2">
        <button onClick={resetExecution} className="inline-flex items-center gap-1 rounded border px-2 py-1 text-xs">
          <RefreshCw className="h-3.5 w-3.5" /> Refresh
        </button>
        <button onClick={downloadLogs} className="inline-flex items-center gap-1 rounded border px-2 py-1 text-xs">
          <Download className="h-3.5 w-3.5" /> Download
        </button>
      </div>
    </section>
  );
}
