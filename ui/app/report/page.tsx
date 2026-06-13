"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import type { RecommendationOut } from "@/lib/types";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { ModeBadge } from "@/components/ModeBadge";

function ReportInner() {
  const router = useRouter();
  const params = useSearchParams();
  const sid = params.get("sid");

  const [rec, setRec] = useState<RecommendationOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exported, setExported] = useState(false);

  useEffect(() => {
    if (!sid) return;
    api.getRecommendation(parseInt(sid))
      .then(setRec)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [sid]);

  const exportMarkdown = () => {
    if (!rec) return;
    const s = rec.summary;
    const md = [
      `# Capacity Recommendation`,
      ``,
      `> **${rec.summary.mode === "estimate_only" ? "⚠ ESTIMATE ONLY" : "✓ VALIDATED"}**`,
      ``,
      `| | |`,
      `|---|---|`,
      `| Model | \`${s.model}\` |`,
      `| GPU | \`${s.gpu}\` |`,
      `| Dtype | \`${s.dtype}\` |`,
      `| Confidence | \`${s.confidence.toUpperCase()}\` |`,
      `| Replicas | **${s.replicas}** |`,
      `| Range | ${s.replicas_low} – ${s.replicas_high} |`,
      `| Binding constraint | \`${s.binding_constraint}\` |`,
      `| Benchmark runs done | ${s.benchmark_runs_done} |`,
    ].join("\n");

    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `recommendation-scenario-${sid}.md`;
    a.click();
    URL.revokeObjectURL(url);
    setExported(true);
  };

  if (!sid) return <p className="text-red-500">Missing scenario ID.</p>;
  if (loading) return <p className="text-gray-500">Loading recommendation…</p>;
  if (error) return <p className="text-red-600 bg-red-50 p-3 rounded">{error}</p>;
  if (!rec) return null;

  const s = rec.summary;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold">Recommendation Report</h1>
        <ModeBadge mode={s.mode} />
      </div>

      <ConfidenceBadge level={rec.confidence} />

      {/* Summary card */}
      <div className="bg-white border rounded-xl p-6">
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-6">
          {[
            { label: "Model", value: s.model },
            { label: "GPU", value: s.gpu },
            { label: "Dtype", value: s.dtype },
            { label: "Recommended replicas", value: s.replicas, big: true },
            { label: "Range", value: `${s.replicas_low} – ${s.replicas_high}` },
            { label: "Binding constraint", value: s.binding_constraint },
          ].map(({ label, value, big }) => (
            <div key={label}>
              <p className="text-xs text-gray-500">{label}</p>
              <p className={`font-semibold ${big ? "text-3xl text-brand-700" : "text-sm"}`}>
                {value}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Validation status */}
      <div className={`rounded-xl border p-4 ${
        s.mode === "validated_by_benchmark"
          ? "bg-green-50 border-green-200"
          : "bg-yellow-50 border-yellow-200"
      }`}>
        <h2 className="text-sm font-semibold mb-1">
          {s.mode === "validated_by_benchmark" ? "✓ Benchmark evidence" : "⚠ No benchmark data"}
        </h2>
        <p className="text-sm">
          {s.benchmark_runs_done === 0
            ? "This recommendation is based on a roofline estimate only. Run the benchmark plan to improve confidence."
            : `${s.benchmark_runs_done} benchmark run(s) completed. Confidence calibrated from live GPU data.`}
        </p>
        {s.mode !== "validated_by_benchmark" && (
          <button
            onClick={() => router.push(`/benchmark-plan?sid=${sid}`)}
            className="mt-3 text-sm text-brand-600 hover:text-brand-700 font-medium"
          >
            → View benchmark plan
          </button>
        )}
      </div>

      {/* Export */}
      <div className="flex gap-3">
        <button
          onClick={() => router.push("/")}
          className="px-4 py-2 border rounded-lg text-sm text-gray-600 hover:bg-gray-50"
        >
          ← New scenario
        </button>
        <button
          onClick={exportMarkdown}
          className="flex-1 bg-brand-600 hover:bg-brand-700 text-white py-2 rounded-lg text-sm font-medium transition-colors"
        >
          {exported ? "✓ Exported" : "Export as Markdown"}
        </button>
      </div>
    </div>
  );
}

export default function ReportPage() {
  return (
    <Suspense fallback={<p className="text-gray-400">Loading…</p>}>
      <ReportInner />
    </Suspense>
  );
}
