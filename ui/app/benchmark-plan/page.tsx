"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import type { BenchmarkPlanOut, BenchmarkRunOut } from "@/lib/types";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { ModeBadge } from "@/components/ModeBadge";
import { CopyButton } from "@/components/CopyButton";

type RunState = {
  runId: number | null;
  status: BenchmarkRunOut["status"] | null;
  error: string | null;
};

const STATUS_COLORS: Record<string, string> = {
  queued:  "bg-gray-100 text-gray-600",
  running: "bg-blue-100 text-blue-700",
  done:    "bg-green-100 text-green-700",
  failed:  "bg-red-100 text-red-700",
};

function BenchmarkPlanInner() {
  const router = useRouter();
  const params = useSearchParams();
  const sid = params.get("sid");

  const [plan, setPlan] = useState<BenchmarkPlanOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // endpoint URL + optional basic auth
  const [endpoint, setEndpoint] = useState("http://localhost:8000/v1/completions");
  const [authUser, setAuthUser] = useState("");
  const [authPass, setAuthPass] = useState("");

  // per-step run state: index → RunState
  const [runs, setRuns] = useState<Record<number, RunState>>({});

  useEffect(() => {
    if (!sid) return;
    api.getBenchmarkPlan(parseInt(sid))
      .then(setPlan)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [sid]);

  // Poll any in-flight runs every 3 s until done/failed
  useEffect(() => {
    const inFlight = Object.entries(runs).filter(
      ([, r]) => r.runId !== null && (r.status === "queued" || r.status === "running")
    );
    if (inFlight.length === 0) return;

    const timer = setInterval(async () => {
      for (const [idxStr, r] of inFlight) {
        if (!r.runId) continue;
        try {
          const updated = await api.getBenchmarkRun(r.runId);
          setRuns((prev) => ({
            ...prev,
            [parseInt(idxStr)]: { ...prev[parseInt(idxStr)], status: updated.status },
          }));
        } catch {
          // ignore transient poll errors
        }
      }
    }, 3000);

    return () => clearInterval(timer);
  }, [runs]);

  const handleRun = async (stepIndex: number) => {
    if (!sid) return;
    setRuns((prev) => ({ ...prev, [stepIndex]: { runId: null, status: "queued", error: null } }));

    // Embed basic auth credentials into the URL if provided
    let effectiveEndpoint = endpoint;
    if (authUser || authPass) {
      try {
        const u = new URL(endpoint);
        u.username = authUser;
        u.password = authPass;
        effectiveEndpoint = u.toString();
      } catch {
        // leave endpoint as-is if URL parsing fails
      }
    }

    try {
      const run = await api.runBenchmark(parseInt(sid), stepIndex, effectiveEndpoint);
      setRuns((prev) => ({ ...prev, [stepIndex]: { runId: run.id, status: run.status, error: null } }));
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setRuns((prev) => ({ ...prev, [stepIndex]: { runId: null, status: "failed", error: msg } }));
    }
  };

  if (!sid) return <p className="text-red-500">Missing scenario ID.</p>;
  if (loading) return <p className="text-gray-500">Generating benchmark plan…</p>;
  if (error) return <p className="text-red-600 bg-red-50 p-3 rounded">{error}</p>;
  if (!plan) return null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold">Benchmark Plan</h1>
        <ModeBadge mode="estimate_only" />
      </div>

      <div className="flex gap-3 flex-wrap">
        <ConfidenceBadge level={plan.confidence} />
        <span className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-gray-100 text-gray-700 border">
          {plan.binding_constraint}
        </span>
      </div>

      {/* Rationale */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-blue-800">
        <strong>Why this ordering:</strong> {plan.rationale}
      </div>

      {/* Endpoint input */}
      <div className="bg-white border rounded-xl p-4 space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          Inference endpoint URL
        </label>
        <p className="text-xs text-gray-500">
          The OpenAI-compatible <code>/v1/completions</code> URL of the running GPU server you want to benchmark.
        </p>
        <input
          type="url"
          value={endpoint}
          onChange={(e) => setEndpoint(e.target.value)}
          placeholder="http://your-gpu-host:8000/v1/completions"
          className="w-full border rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-brand-500"
        />
        <details className="mt-1">
          <summary className="text-xs text-gray-400 cursor-pointer select-none hover:text-gray-600">
            Basic auth (optional)
          </summary>
          <div className="mt-2 flex gap-2">
            <input
              type="text"
              value={authUser}
              onChange={(e) => setAuthUser(e.target.value)}
              placeholder="Username"
              autoComplete="username"
              className="flex-1 border rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
            />
            <input
              type="password"
              value={authPass}
              onChange={(e) => setAuthPass(e.target.value)}
              placeholder="Password"
              autoComplete="current-password"
              className="flex-1 border rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
            />
          </div>
          <p className="text-xs text-gray-400 mt-1">
            Credentials are embedded in the URL sent to the backend — never stored.
          </p>
        </details>
      </div>

      {/* Steps */}
      <div className="space-y-3">
        {plan.steps.map((step, i) => {
          const run = runs[i];
          const isActive = run?.status === "queued" || run?.status === "running";
          return (
            <div key={i} className="bg-white border rounded-xl p-4">
              <div className="flex items-start justify-between gap-3 mb-2">
                <div>
                  <span className="text-xs text-gray-400 font-mono mr-2">#{step.priority}</span>
                  <span className="font-medium text-sm">{step.label}</span>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {run?.status && (
                    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${STATUS_COLORS[run.status]}`}>
                      {run.status}
                    </span>
                  )}
                  <CopyButton text={step.command} />
                  <button
                    onClick={() => handleRun(i)}
                    disabled={isActive || !endpoint}
                    className="text-xs px-3 py-1 rounded-lg bg-brand-600 hover:bg-brand-700 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    {isActive ? "Running…" : "Run"}
                  </button>
                </div>
              </div>
              <p className="text-xs text-gray-500 mb-2">{step.purpose}</p>
              <pre className="text-xs bg-gray-50 border rounded p-3 overflow-x-auto text-gray-700 whitespace-pre-wrap break-all">
                {step.command}
              </pre>
              {run?.error && (
                <p className="text-xs text-red-600 mt-2">{run.error}</p>
              )}
              <p className="text-xs text-gray-400 mt-2">
                Collapses: <em>{step.collapses_confidence_on}</em>
              </p>
            </div>
          );
        })}
      </div>

      {/* All commands block */}
      <div className="bg-white border rounded-xl p-4">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-sm font-medium">All commands</h2>
          <CopyButton text={plan.steps.map((s) => s.command).join("\n")} />
        </div>
        <pre className="text-xs bg-gray-50 p-3 rounded overflow-x-auto text-gray-600 whitespace-pre-wrap break-all">
          {plan.steps.map((s) => s.command).join("\n")}
        </pre>
      </div>

      {/* Navigation */}
      <div className="flex gap-3">
        <button
          onClick={() => router.push(`/estimate?sid=${sid}`)}
          className="px-4 py-2 border rounded-lg text-sm text-gray-600 hover:bg-gray-50"
        >
          ← Estimate
        </button>
        <button
          onClick={() => router.push(`/report?sid=${sid}`)}
          className="flex-1 bg-brand-600 hover:bg-brand-700 text-white py-2 rounded-lg text-sm font-medium transition-colors"
        >
          View Recommendation →
        </button>
      </div>
    </div>
  );
}

export default function BenchmarkPlanPage() {
  return (
    <Suspense fallback={<p className="text-gray-400">Loading…</p>}>
      <BenchmarkPlanInner />
    </Suspense>
  );
}
