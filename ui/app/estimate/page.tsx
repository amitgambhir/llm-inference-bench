"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { api } from "@/lib/api";
import type { EstimateOut } from "@/lib/types";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { ModeBadge } from "@/components/ModeBadge";
import { ReplicaRangeChart } from "@/components/ReplicaRangeChart";

function EstimateResultsInner() {
  const router = useRouter();
  const params = useSearchParams();
  const sid = params.get("sid");

  const [estimate, setEstimate] = useState<EstimateOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sid) return;
    api.runEstimate(parseInt(sid))
      .then(setEstimate)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [sid]);

  if (!sid) return <p className="text-red-500">Missing scenario ID.</p>;
  if (loading) return <p className="text-gray-500">Running estimate…</p>;
  if (error) return <p className="text-red-600 bg-red-50 p-3 rounded">{error}</p>;
  if (!estimate) return null;

  const p = estimate.payload;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Estimate Results</h1>
        <ModeBadge mode="estimate_only" />
      </div>

      {/* Confidence + binding constraint */}
      <div className="flex gap-3 flex-wrap">
        <ConfidenceBadge level={estimate.confidence} />
        <span className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-gray-100 text-gray-700 border">
          {estimate.binding_constraint}
        </span>
      </div>

      {/* Key numbers */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Recommended replicas", value: estimate.replicas },
          { label: "Range", value: `${estimate.replicas_low} – ${estimate.replicas_high}` },
          { label: "Predicted TTFT", value: `${p.ttft_ms?.toFixed(0) ?? "–"} ms` },
        ].map(({ label, value }) => (
          <div key={label} className="bg-white rounded-xl border p-4 text-center">
            <p className="text-2xl font-bold text-brand-700">{value}</p>
            <p className="text-xs text-gray-500 mt-1">{label}</p>
          </div>
        ))}
      </div>

      {/* Replica range chart */}
      <div className="bg-white rounded-xl border p-4">
        <h2 className="text-sm font-medium mb-3 text-gray-700">Replica range</h2>
        <ReplicaRangeChart
          replicas={estimate.replicas}
          replicasLow={estimate.replicas_low}
          replicasHigh={estimate.replicas_high}
          confidence={estimate.confidence}
        />
      </div>

      {/* Workload stats */}
      <div className="bg-white rounded-xl border p-4">
        <h2 className="text-sm font-medium mb-3 text-gray-700">Workload stats</h2>
        <dl className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
          {[
            ["Avg RPS", p.avg_rps?.toFixed(2)],
            ["Peak RPS", p.peak_rps?.toFixed(2)],
            ["Prefill ceiling/GPU", `${p.prefill_tps_gpu?.toFixed(0)} tok/s`],
            ["Decode ceiling/GPU", `${p.decode_tps_gpu?.toFixed(0)} tok/s`],
            ["Max concurrent seqs/replica", p.max_concurrent_seqs],
            ["MFU used", `${(p.mfu_used * 100).toFixed(1)}%`],
          ].map(([k, v]) => (
            <div key={k as string} className="flex justify-between border-b py-1">
              <dt className="text-gray-500">{k}</dt>
              <dd className="font-mono">{v}</dd>
            </div>
          ))}
        </dl>
      </div>

      {/* Warnings */}
      {p.warnings?.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-yellow-800 mb-2">Warnings</h2>
          <ul className="space-y-1">
            {p.warnings.map((w, i) => (
              <li key={i} className="text-sm text-yellow-700">⚠ {w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Navigation */}
      <div className="flex gap-3">
        <button
          onClick={() => router.push("/")}
          className="px-4 py-2 border rounded-lg text-sm text-gray-600 hover:bg-gray-50"
        >
          ← New scenario
        </button>
        <button
          onClick={() => router.push(`/benchmark-plan?sid=${sid}`)}
          className="flex-1 bg-brand-600 hover:bg-brand-700 text-white py-2 rounded-lg text-sm font-medium transition-colors"
        >
          Generate Benchmark Plan →
        </button>
      </div>
    </div>
  );
}

export default function EstimateResults() {
  return (
    <Suspense fallback={<p className="text-gray-400">Loading…</p>}>
      <EstimateResultsInner />
    </Suspense>
  );
}
