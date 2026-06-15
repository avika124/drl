"use client";

import { useWorkflowStore } from "@/lib/store";
import { ParameterConfig, WorkflowNode } from "@/lib/types";

const ORDER: WorkflowNode[] = ["n1", "n2", "n3", "n4", "n5", "n6", "n7"];

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export function useWorkflow() {
  const store = useWorkflowStore();

  const exportConfig = () => {
    const blob = new Blob([JSON.stringify(store.parameters, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "drl-config.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const importConfigFile = async (file: File) => {
    const text = await file.text();
    const parsed = JSON.parse(text) as ParameterConfig;
    store.importConfig(parsed);
    store.addLog("Imported configuration JSON.");
  };

  const runAll = async () => {
    store.resetExecution();
    store.setRunning(true);
    store.addLog("Starting M1 -> M2 workflow execution...");

    for (let i = 0; i < ORDER.length; i++) {
      const node = ORDER[i];
      store.setNodeStatus(node, "running");
      store.setSelectedNode(node);
      store.setProgress(Math.round((i / ORDER.length) * 100));
      store.addLog(`Executing ${node}...`);
      await wait(350 + i * 60);
      const ms = Math.round(4 + Math.random() * 80);
      store.setNodeStatus(node, "success", ms);
      store.addLog(`${node} completed in ${ms}ms.`);
    }

    const trainRes = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parameters: store.parameters }),
    });
    const trainJson = await trainRes.json();
    store.addLog(`Checkpoint saved: ${trainJson.checkpoint}`);

    const optimizeRes = await fetch("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        campaign_id: "cmp_7f3a",
        state: Array.from({ length: 42 }, (_, idx) => Number((0.01 * (idx + 1)).toFixed(3))),
        constraints: { target_cpa: 25, min_roas: 2.0, daily_budget: 2000 },
        parameters: store.parameters,
      }),
    });
    const optimizeJson = await optimizeRes.json();
    store.setResult(optimizeJson);
    store.setProgress(100);
    store.addLog(`Optimization complete in ${optimizeJson.execution_time_ms}ms.`);
    store.setRunning(false);
  };

  return {
    ...store,
    runAll,
    exportConfig,
    importConfigFile,
  };
}
