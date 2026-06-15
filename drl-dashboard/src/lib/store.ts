"use client";

import { create } from "zustand";
import { DEFAULT_PARAMS, PRESETS } from "./simulations";
import { LogEntry, NodeStatus, OptimizationResult, ParameterConfig, PresetName, WorkflowNode } from "./types";

type NodeStatusMap = Record<WorkflowNode, NodeStatus>;

const baseNodeStatuses: NodeStatusMap = {
  n1: "pending",
  n2: "pending",
  n3: "pending",
  n4: "pending",
  n5: "pending",
  n6: "pending",
  n7: "pending",
};

interface WorkflowStore {
  parameters: ParameterConfig;
  baselineParameters: ParameterConfig;
  modifiedCount: number;
  selectedNode: WorkflowNode | null;
  showDetailPanel: boolean;
  isRunning: boolean;
  progress: number;
  nodeStatuses: NodeStatusMap;
  nodeTimes: Partial<Record<WorkflowNode, number>>;
  executionLogs: LogEntry[];
  lastResult: OptimizationResult | null;
  setParameter: <K extends keyof ParameterConfig>(key: K, value: ParameterConfig[K]) => void;
  loadPreset: (preset: PresetName) => void;
  setSelectedNode: (node: WorkflowNode | null) => void;
  setNodeStatus: (nodeId: WorkflowNode, status: NodeStatus, ms?: number) => void;
  addLog: (message: string, level?: LogEntry["level"]) => void;
  setRunning: (value: boolean) => void;
  setProgress: (value: number) => void;
  setResult: (result: OptimizationResult | null) => void;
  resetExecution: () => void;
  importConfig: (payload: ParameterConfig) => void;
}

const countModified = (a: ParameterConfig, b: ParameterConfig) =>
  (Object.keys(a) as Array<keyof ParameterConfig>).filter((key) => a[key] !== b[key]).length;

export const useWorkflowStore = create<WorkflowStore>((set, get) => ({
  parameters: DEFAULT_PARAMS,
  baselineParameters: DEFAULT_PARAMS,
  modifiedCount: 0,
  selectedNode: null,
  showDetailPanel: true,
  isRunning: false,
  progress: 0,
  nodeStatuses: baseNodeStatuses,
  nodeTimes: {},
  executionLogs: [],
  lastResult: null,
  setParameter: (key, value) =>
    set((state) => {
      const next = { ...state.parameters, [key]: value };
      return {
        parameters: next,
        modifiedCount: countModified(next, state.baselineParameters),
      };
    }),
  loadPreset: (preset) =>
    set(() => {
      const params = PRESETS[preset];
      return {
        parameters: params,
        baselineParameters: params,
        modifiedCount: 0,
      };
    }),
  setSelectedNode: (node) => set({ selectedNode: node }),
  setNodeStatus: (nodeId, status, ms) =>
    set((state) => ({
      nodeStatuses: { ...state.nodeStatuses, [nodeId]: status },
      nodeTimes: ms ? { ...state.nodeTimes, [nodeId]: ms } : state.nodeTimes,
    })),
  addLog: (message, level = "info") =>
    set((state) => ({
      executionLogs: [
        ...state.executionLogs.slice(-80),
        { id: crypto.randomUUID(), at: new Date().toLocaleTimeString(), level, message },
      ],
    })),
  setRunning: (value) => set({ isRunning: value }),
  setProgress: (value) => set({ progress: value }),
  setResult: (result) => set({ lastResult: result }),
  resetExecution: () =>
    set({
      progress: 0,
      nodeStatuses: baseNodeStatuses,
      nodeTimes: {},
      executionLogs: [],
      lastResult: null,
    }),
  importConfig: (payload) => {
    const merged = { ...get().parameters, ...payload, state_dim: 42 };
    set((state) => ({
      parameters: merged,
      modifiedCount: countModified(merged, state.baselineParameters),
    }));
  },
}));
