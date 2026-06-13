// API response types — mirrors api/schemas.py

export interface ScenarioCreate {
  name: string;
  model_name: string;
  gpu_name: string;
  dtype: string;
  requests_per_day: number;
  peak_multiplier: number;
  isl: number;
  osl: number;
  ttft_slo_ms: number;
  tp: number;
  traffic_class: string;
  gpu_mem_util: number;
  runtime: string;
}

export interface ScenarioOut extends ScenarioCreate {
  id: number;
  created_at: string;
}

export interface EstimateOut {
  id: number;
  scenario_id: number;
  replicas: number;
  replicas_low: number;
  replicas_high: number;
  binding_constraint: string;
  confidence: "high" | "medium" | "low";
  payload: {
    avg_rps: number;
    peak_rps: number;
    ttft_ms: number;
    max_concurrent_seqs: number;
    mfu_used: number;
    prefill_tps_gpu: number;
    decode_tps_gpu: number;
    warnings: string[];
    assumptions: string[];
  };
  created_at: string;
}

export interface BenchmarkStepOut {
  priority: number;
  label: string;
  command: string;
  purpose: string;
  collapses_confidence_on: string;
}

export interface BenchmarkPlanOut {
  steps: BenchmarkStepOut[];
  confidence: string;
  binding_constraint: string;
  rationale: string;
}

export interface BenchmarkRunOut {
  id: number;
  scenario_id: number;
  tag: string;
  command: string;
  status: "queued" | "running" | "done" | "failed";
  result_path: string | null;
  exit_code: number | null;
  created_at: string;
}

export interface RecommendationOut {
  id: number;
  scenario_id: number;
  summary: {
    mode: string;
    replicas: number;
    replicas_low: number;
    replicas_high: number;
    binding_constraint: string;
    confidence: string;
    benchmark_runs_done: number;
    model: string;
    gpu: string;
    dtype: string;
  };
  confidence: string;
  created_at: string;
}

// GPU and model catalog entries (fetched from API if needed)
export const CATALOG_GPUS: { key: string; label: string }[] = [
  { key: "h100_sxm", label: "NVIDIA H100 SXM (80 GB)" },
  { key: "h200_sxm", label: "NVIDIA H200 SXM (141 GB)" },
  { key: "a100_80gb_sxm", label: "NVIDIA A100 SXM (80 GB)" },
  { key: "l40s", label: "NVIDIA L40S (48 GB)" },
  { key: "l4", label: "NVIDIA L4 (24 GB)" },
];

export const CATALOG_MODELS: { key: string; label: string }[] = [
  { key: "llama-3.1-8b", label: "Llama 3.1 8B" },
  { key: "llama-3.1-70b", label: "Llama 3.1 70B" },
  { key: "gpt-oss-20b", label: "GPT-OSS 20B (MoE)" },
];

export const DTYPES = ["bf16", "fp16", "fp8", "mxfp4", "int8"];

export const TRAFFIC_CLASSES = [
  { key: "realtime", label: "Realtime (1.40× headroom)" },
  { key: "mixed", label: "Mixed (1.25× headroom)" },
  { key: "batch", label: "Batch (1.10× headroom)" },
];
