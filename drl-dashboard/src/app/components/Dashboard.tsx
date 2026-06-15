"use client";

import { ResultsDisplay } from "./ResultsDisplay";
import { Sidebar } from "./Sidebar";
import { WorkflowVisualizer } from "./WorkflowVisualizer";

export function Dashboard() {
  return (
    <main className="mx-auto grid w-full max-w-[1500px] gap-3 p-3 lg:grid-cols-[320px_1fr_420px]">
      <aside className="lg:col-span-1">
        <Sidebar />
      </aside>
      <section className="lg:col-span-1">
        <WorkflowVisualizer />
      </section>
      <section className="lg:col-span-1">
        <ResultsDisplay />
      </section>
    </main>
  );
}
