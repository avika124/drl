"use client";

import { ExecutionMonitor } from "./ExecutionMonitor";
import { ParameterPanel } from "./ParameterPanel";

export function Sidebar() {
  return (
    <div className="space-y-3">
      <ParameterPanel />
      <ExecutionMonitor />
    </div>
  );
}
