"""Pydantic request/response schemas for the capacity planner API.

These are separate from the SQLAlchemy ORM models so the API contract can
evolve independently of the persistence layer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Scenarios ─────────────────────────────────────────────────────────────────


class ScenarioCreate(BaseModel):
    name: str
    model_name: str
    gpu_name: str = "h100_sxm"
    dtype: str = "bf16"
    requests_per_day: int = Field(..., gt=0)
    peak_multiplier: float = Field(3.0, gt=0)
    isl: int = Field(..., gt=0)
    osl: int = Field(..., gt=0)
    ttft_slo_ms: float = Field(2000.0, gt=0)
    tp: int = Field(1, ge=1)
    traffic_class: str = "realtime"
    gpu_mem_util: float = Field(0.90, gt=0, le=1.0)
    runtime: Optional[str] = "vllm"


class ScenarioOut(ScenarioCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Estimates ─────────────────────────────────────────────────────────────────


class EstimateRequest(BaseModel):
    scenario_id: int


class EstimateOut(BaseModel):
    id: int
    scenario_id: int
    replicas: int
    replicas_low: int
    replicas_high: int
    binding_constraint: str
    confidence: str
    payload: Dict[str, Any]
    created_at: datetime


# ── Benchmark plan ─────────────────────────────────────────────────────────────


class BenchmarkPlanRequest(BaseModel):
    scenario_id: int


class BenchmarkStepOut(BaseModel):
    priority: int
    label: str
    command: str
    purpose: str
    collapses_confidence_on: str


class BenchmarkPlanOut(BaseModel):
    steps: List[BenchmarkStepOut]
    confidence: str
    binding_constraint: str
    rationale: str


# ── Benchmark runs ─────────────────────────────────────────────────────────────


class RunBenchmarkRequest(BaseModel):
    scenario_id: int
    endpoint: str = "http://localhost:8000"
    step_index: int = Field(0, ge=0)
    output_dir: Optional[str] = None   # override results dir (useful in tests)


class BenchmarkRunOut(BaseModel):
    id: int
    scenario_id: int
    tag: str
    command: str
    status: str
    stdout: Optional[str] = None
    exit_code: Optional[int] = None
    result_path: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Ingest ────────────────────────────────────────────────────────────────────


class IngestOut(BaseModel):
    anchor: Dict[str, Any]
    impact: List[str]


# ── Recommendation ────────────────────────────────────────────────────────────


class RecommendationOut(BaseModel):
    id: int
    scenario_id: int
    summary: Dict[str, Any]
    confidence: str
    created_at: datetime


# ── Report ────────────────────────────────────────────────────────────────────


class ReportOut(BaseModel):
    id: int
    scenario_id: int
    markdown_content: str
    created_at: datetime

    model_config = {"from_attributes": True}
