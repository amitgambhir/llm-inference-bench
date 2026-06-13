import type {
  BenchmarkPlanOut,
  BenchmarkRunOut,
  EstimateOut,
  RecommendationOut,
  ScenarioCreate,
  ScenarioOut,
} from "./types";

const BASE = "/api";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail?: string }).detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail?: string }).detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export const api = {
  createScenario: (body: ScenarioCreate) =>
    post<ScenarioOut>("/scenarios", body),

  runEstimate: (scenarioId: number) =>
    post<EstimateOut>("/estimate", { scenario_id: scenarioId }),

  getBenchmarkPlan: (scenarioId: number) =>
    post<BenchmarkPlanOut>("/benchmark-plan", { scenario_id: scenarioId }),

  runBenchmark: (scenarioId: number, stepIndex: number, endpoint: string) =>
    post<BenchmarkRunOut>("/benchmarks/run", {
      scenario_id: scenarioId,
      step_index: stepIndex,
      endpoint,
    }),

  getBenchmarkRun: (runId: number) =>
    get<BenchmarkRunOut>(`/benchmarks/${runId}`),

  getRecommendation: (scenarioId: number) =>
    get<RecommendationOut>(`/scenarios/${scenarioId}/recommendation`),
};
