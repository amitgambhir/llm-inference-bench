"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import {
  CATALOG_GPUS,
  CATALOG_MODELS,
  DTYPES,
  TRAFFIC_CLASSES,
  type ScenarioCreate,
} from "@/lib/types";

// ── Custom model form ─────────────────────────────────────────────────────────

function CustomModelFields({
  onChange,
}: {
  onChange: (spec: Record<string, unknown> | null) => void;
}) {
  const [mode, setMode] = useState<"rough" | "full">("rough");
  const [params, setParams] = useState("");
  const [activeParams, setActiveParams] = useState("");

  const handleChange = () => {
    if (!params) { onChange(null); return; }
    const totalParams = parseFloat(params) * 1e9;
    const spec: Record<string, unknown> = { total_params: totalParams, is_moe: false };
    if (activeParams) spec.active_params = parseFloat(activeParams) * 1e9;
    onChange(spec);
  };

  return (
    <div className="mt-2 p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm">
      <p className="text-amber-800 font-medium mb-2">Custom model — geometry estimated from params</p>
      <div className="flex gap-2 mb-3">
        <button
          type="button"
          onClick={() => setMode("rough")}
          className={`px-2 py-1 rounded text-xs ${mode === "rough" ? "bg-amber-600 text-white" : "bg-white border"}`}
        >
          Param count only
        </button>
        <button
          type="button"
          onClick={() => setMode("full")}
          className={`px-2 py-1 rounded text-xs ${mode === "full" ? "bg-amber-600 text-white" : "bg-white border"}`}
        >
          Full spec
        </button>
      </div>
      {mode === "rough" && (
        <div className="flex gap-3">
          <label className="flex flex-col gap-1 flex-1">
            <span className="text-gray-600">Total params (B)</span>
            <input
              type="number"
              className="border rounded px-2 py-1"
              placeholder="e.g. 70"
              value={params}
              onChange={(e) => { setParams(e.target.value); handleChange(); }}
            />
          </label>
          <label className="flex flex-col gap-1 flex-1">
            <span className="text-gray-600">Active params (B) — MoE</span>
            <input
              type="number"
              className="border rounded px-2 py-1"
              placeholder="Leave blank if dense"
              value={activeParams}
              onChange={(e) => { setActiveParams(e.target.value); handleChange(); }}
            />
          </label>
        </div>
      )}
      <p className="text-amber-700 text-xs mt-2">
        ⚠ Estimated geometry — confidence will be capped at MEDIUM.
      </p>
    </div>
  );
}

// ── Scenario Builder (screen 1) ───────────────────────────────────────────────

export default function ScenarioBuilder() {
  const router = useRouter();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelMode, setModelMode] = useState<"catalog" | "custom">("catalog");

  const [form, setForm] = useState<ScenarioCreate>({
    name: "my-deployment",
    model_name: "llama-3.1-8b",
    gpu_name: "h100_sxm",
    dtype: "bf16",
    requests_per_day: 1_000_000,
    peak_multiplier: 3,
    isl: 2048,
    osl: 256,
    ttft_slo_ms: 2000,
    tp: 1,
    traffic_class: "realtime",
    gpu_mem_util: 0.9,
    runtime: "vllm",
  });

  const set = (k: keyof ScenarioCreate, v: unknown) =>
    setForm((f) => ({ ...f, [k]: v }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const scenario = await api.createScenario(form);
      router.push(`/estimate?sid=${scenario.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-1">Scenario Builder</h1>
      <p className="text-gray-500 text-sm mb-6">
        Describe your workload. We'll size replicas, estimate cost, and generate a benchmark plan.
      </p>

      <form onSubmit={handleSubmit} className="space-y-6">

        {/* ── Name ── */}
        <div>
          <label className="block text-sm font-medium mb-1">Scenario name</label>
          <input
            className="w-full border rounded-lg px-3 py-2 text-sm"
            value={form.name}
            onChange={(e) => set("name", e.target.value)}
            required
          />
        </div>

        {/* ── GPU (dropdown only) ── */}
        <div>
          <label className="block text-sm font-medium mb-1">
            GPU <span className="text-gray-400 font-normal">(curated catalog — new GPUs added by maintainer)</span>
          </label>
          <select
            className="w-full border rounded-lg px-3 py-2 text-sm"
            value={form.gpu_name}
            onChange={(e) => set("gpu_name", e.target.value)}
          >
            {CATALOG_GPUS.map((g) => (
              <option key={g.key} value={g.key}>{g.label}</option>
            ))}
          </select>
        </div>

        {/* ── Model (dropdown + custom) ── */}
        <div>
          <label className="block text-sm font-medium mb-1">Model</label>
          <div className="flex gap-2 mb-2">
            <button
              type="button"
              onClick={() => setModelMode("catalog")}
              className={`px-3 py-1 rounded text-sm ${modelMode === "catalog" ? "bg-brand-600 text-white" : "bg-white border"}`}
            >
              Catalog
            </button>
            <button
              type="button"
              onClick={() => setModelMode("custom")}
              className={`px-3 py-1 rounded text-sm ${modelMode === "custom" ? "bg-brand-600 text-white" : "bg-white border"}`}
            >
              Custom
            </button>
          </div>
          {modelMode === "catalog" ? (
            <select
              className="w-full border rounded-lg px-3 py-2 text-sm"
              value={form.model_name}
              onChange={(e) => set("model_name", e.target.value)}
            >
              {CATALOG_MODELS.map((m) => (
                <option key={m.key} value={m.key}>{m.label}</option>
              ))}
            </select>
          ) : (
            <>
              <input
                className="w-full border rounded-lg px-3 py-2 text-sm"
                placeholder="e.g. my-custom-70b"
                value={form.model_name}
                onChange={(e) => set("model_name", e.target.value)}
              />
              <CustomModelFields onChange={(spec) => {
                if (spec) set("model_name", JSON.stringify(spec));
              }} />
            </>
          )}
        </div>

        {/* ── Dtype + TP ── */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Serving dtype</label>
            <select
              className="w-full border rounded-lg px-3 py-2 text-sm"
              value={form.dtype}
              onChange={(e) => set("dtype", e.target.value)}
            >
              {DTYPES.map((d) => <option key={d}>{d}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Tensor parallel</label>
            <input
              type="number" min={1} max={8}
              className="w-full border rounded-lg px-3 py-2 text-sm"
              value={form.tp}
              onChange={(e) => set("tp", parseInt(e.target.value))}
            />
          </div>
        </div>

        {/* ── Traffic ── */}
        <fieldset className="border rounded-lg p-4">
          <legend className="text-sm font-medium px-1">Traffic</legend>
          <div className="grid grid-cols-2 gap-4 mt-2">
            <div>
              <label className="block text-sm mb-1">Requests/day</label>
              <input
                type="number" min={1}
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.requests_per_day}
                onChange={(e) => set("requests_per_day", parseInt(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">Peak multiplier</label>
              <input
                type="number" min={1} step={0.5}
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.peak_multiplier}
                onChange={(e) => set("peak_multiplier", parseFloat(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">ISL (input tokens)</label>
              <input
                type="number" min={1}
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.isl}
                onChange={(e) => set("isl", parseInt(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">OSL (output tokens)</label>
              <input
                type="number" min={1}
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.osl}
                onChange={(e) => set("osl", parseInt(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">TTFT SLO (ms)</label>
              <input
                type="number" min={100}
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.ttft_slo_ms}
                onChange={(e) => set("ttft_slo_ms", parseFloat(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">Traffic class</label>
              <select
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.traffic_class}
                onChange={(e) => set("traffic_class", e.target.value)}
              >
                {TRAFFIC_CLASSES.map((tc) => (
                  <option key={tc.key} value={tc.key}>{tc.label}</option>
                ))}
              </select>
            </div>
          </div>
        </fieldset>

        {error && (
          <p className="text-red-600 text-sm bg-red-50 border border-red-200 rounded p-3">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={submitting}
          className="w-full bg-brand-600 hover:bg-brand-700 text-white font-medium py-3 rounded-lg disabled:opacity-50 transition-colors"
        >
          {submitting ? "Creating scenario…" : "Build Estimate →"}
        </button>
      </form>
    </div>
  );
}
