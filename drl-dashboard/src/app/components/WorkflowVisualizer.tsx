"use client";

import { line, curveMonotoneX } from "d3-shape";
import { useMemo } from "react";
import { useWorkflow } from "@/hooks/useWorkflow";
import { WorkflowNode } from "@/lib/types";

const nodes: Array<{ id: WorkflowNode; label: string; lane: "m1" | "m2"; x: number; y: number }> = [
  { id: "n1", label: "Campaign Data", lane: "m1", x: 80, y: 80 },
  { id: "n2", label: "Replay Buffer", lane: "m1", x: 250, y: 80 },
  { id: "n3", label: "SAC Training", lane: "m1", x: 430, y: 80 },
  { id: "n4", label: "Checkpoint", lane: "m1", x: 610, y: 80 },
  { id: "n5", label: "Load SAC", lane: "m2", x: 160, y: 250 },
  { id: "n6", label: "Safety Guardrails", lane: "m2", x: 360, y: 250 },
  { id: "n7", label: "Hybrid Optimizer", lane: "m2", x: 580, y: 250 },
];

const edges: Array<[WorkflowNode, WorkflowNode]> = [
  ["n1", "n2"],
  ["n2", "n3"],
  ["n3", "n4"],
  ["n4", "n5"],
  ["n5", "n6"],
  ["n6", "n7"],
];

export function WorkflowVisualizer() {
  const { nodeStatuses, selectedNode, setSelectedNode, progress } = useWorkflow();

  const positions = useMemo(
    () => Object.fromEntries(nodes.map((node) => [node.id, { x: node.x, y: node.y }])) as Record<WorkflowNode, { x: number; y: number }>,
    []
  );

  const edgePath = (from: WorkflowNode, to: WorkflowNode) => {
    const a = positions[from];
    const b = positions[to];
    const p = line<{ x: number; y: number }>()
      .curve(curveMonotoneX)
      .x((d) => d.x)
      .y((d) => d.y);
    return p([
      { x: a.x + 52, y: a.y + 18 },
      { x: (a.x + b.x) / 2, y: a.y + 18 },
      { x: b.x - 52, y: b.y + 18 },
    ]);
  };

  const bgByStatus = (id: WorkflowNode) => {
    const status = nodeStatuses[id];
    if (status === "running") return "bg-blue-100 border-blue-500";
    if (status === "success") return "bg-emerald-100 border-emerald-500";
    if (status === "error") return "bg-red-100 border-red-500";
    return "bg-white border-slate-300";
  };

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3">
      <h2 className="mb-2 text-sm font-semibold">Workflow Visualizer</h2>
      <p className="mb-3 text-xs text-slate-600">M1 Training Pipeline → M2 Inference Pipeline</p>
      <div className="mb-3 h-2 w-full overflow-hidden rounded bg-slate-100">
        <div className="h-full bg-blue-600 transition-all" style={{ width: `${progress}%` }} />
      </div>

      <div className="relative h-[340px] overflow-x-auto rounded border border-slate-200 bg-slate-50">
        <svg className="absolute left-0 top-0 h-full w-full" viewBox="0 0 720 340" aria-hidden>
          {edges.map(([from, to]) => (
            <path key={`${from}-${to}`} d={edgePath(from, to) ?? ""} fill="none" stroke="#94a3b8" strokeWidth={2} strokeDasharray="6 4" />
          ))}
        </svg>

        <div className="absolute left-2 top-2 rounded bg-blue-50 px-2 py-1 text-xs font-semibold text-blue-700">M1 — Training</div>
        <div className="absolute left-2 top-[173px] rounded bg-violet-50 px-2 py-1 text-xs font-semibold text-violet-700">M2 — Inference</div>

        {nodes.map((node) => (
          <button
            key={node.id}
            onClick={() => setSelectedNode(node.id)}
            className={`absolute w-[110px] rounded-md border p-2 text-left text-xs shadow-sm ${bgByStatus(node.id)} ${
              selectedNode === node.id ? "ring-2 ring-blue-400" : ""
            }`}
            style={{ left: node.x - 55, top: node.y }}
          >
            <div className="font-semibold">{node.id.toUpperCase()}</div>
            <div className="text-slate-700">{node.label}</div>
          </button>
        ))}
      </div>
    </section>
  );
}
