"""FastAPI application for the LLM Inference Capacity Planner.

Factory pattern (`create_app`) lets tests inject an isolated SQLite URL and
a temp results/anchors directory without monkeypatching module globals.

Start with:
  uvicorn api.main:app --reload
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from api.db import (
    BenchmarkRunRow,
    CapacityEstimateRow,
    RecommendationRow,
    ReportRow,
    ScenarioRow,
    make_engine_and_session,
)
from api.jobs import run_job
from api.schemas import (
    BenchmarkPlanOut,
    BenchmarkPlanRequest,
    BenchmarkRunOut,
    BenchmarkStepOut,
    EstimateOut,
    EstimateRequest,
    IngestOut,
    RecommendationOut,
    ReportOut,
    RunBenchmarkRequest,
    ScenarioCreate,
    ScenarioOut,
)


def _get_db(request: Request):
    """Per-request session from the app-state factory (no module globals)."""
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def create_app(
    db_url: str = "sqlite:///./capacity_planner.db",
    results_dir: str = "results/real",
    anchors_file: Optional[str] = None,
) -> FastAPI:
    """App factory — call with test-specific paths to get an isolated instance."""
    app = FastAPI(title="LLM Inference Capacity Planner", version="0.4.0")

    # CORS — allow the frontend origin (set ALLOWED_ORIGIN in production)
    allowed_origins = os.getenv("ALLOWED_ORIGIN", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine, session_factory = make_engine_and_session(db_url)
    app.state.db_url = db_url
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.results_dir = Path(results_dir)
    app.state.anchors_file = Path(anchors_file) if anchors_file else None

    # TODO: SSO middleware (OAuth2 / OIDC bearer token) goes here for
    # multi-tenant deployments.  Local mode runs without auth.

    # ── POST /scenarios ────────────────────────────────────────────────────

    @app.post("/scenarios", response_model=ScenarioOut, status_code=201)
    def create_scenario(body: ScenarioCreate, db: Session = Depends(_get_db)):
        row = ScenarioRow(
            name=body.name,
            model_name=body.model_name,
            gpu_name=body.gpu_name,
            dtype=body.dtype,
            requests_per_day=body.requests_per_day,
            peak_multiplier=body.peak_multiplier,
            isl=body.isl,
            osl=body.osl,
            ttft_slo_ms=body.ttft_slo_ms,
            tp=body.tp,
            traffic_class=body.traffic_class,
            gpu_mem_util=body.gpu_mem_util,
            runtime=body.runtime,
            created_at=datetime.now(timezone.utc),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    # ── POST /estimate ─────────────────────────────────────────────────────

    @app.post("/estimate", response_model=EstimateOut, status_code=201)
    def run_estimate(body: EstimateRequest, db: Session = Depends(_get_db)):
        scenario = db.get(ScenarioRow, body.scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")

        from planner.capacity import plan
        from planner.catalog import CatalogError, get_gpu, resolve_model

        try:
            model = resolve_model(scenario.model_name)
            gpu = get_gpu(scenario.gpu_name)
        except CatalogError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        try:
            est = plan(
                requests_per_day=scenario.requests_per_day,
                peak_multiplier=scenario.peak_multiplier,
                isl=scenario.isl,
                osl=scenario.osl,
                ttft_slo_ms=scenario.ttft_slo_ms,
                model=model,
                gpu=gpu,
                dtype=scenario.dtype,
                tp=scenario.tp,
                traffic_class=scenario.traffic_class,
                gpu_mem_util=scenario.gpu_mem_util,
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        payload = {
            "avg_rps": est.traffic.avg_rps,
            "peak_rps": est.traffic.peak_rps,
            "input_tps_peak": est.traffic.input_tps_peak,
            "output_tps_peak": est.traffic.output_tps_peak,
            "prefill_tps_gpu": est.prefill_tps_gpu,
            "decode_tps_gpu": est.decode_tps_gpu,
            "ttft_ms": est.ttft_estimate.ttft_ms,
            "max_concurrent_seqs": est.kv_budget.max_concurrent_seqs,
            "mfu_used": est.mfu_used,
            "bw_eff_used": est.bw_eff_used,
            "warnings": est.warnings,
            "assumptions": est.assumptions,
        }

        row = CapacityEstimateRow(
            scenario_id=scenario.id,
            replicas=est.replicas,
            replicas_low=est.replicas_low,
            replicas_high=est.replicas_high,
            binding_constraint=est.binding_constraint,
            confidence=est.confidence,
            payload=json.dumps(payload),
            created_at=datetime.now(timezone.utc),
        )
        db.add(row)
        db.commit()
        db.refresh(row)

        return EstimateOut(
            id=row.id,
            scenario_id=row.scenario_id,
            replicas=row.replicas,
            replicas_low=row.replicas_low,
            replicas_high=row.replicas_high,
            binding_constraint=row.binding_constraint,
            confidence=row.confidence,
            payload=payload,
            created_at=row.created_at,
        )

    # ── POST /benchmark-plan ───────────────────────────────────────────────

    @app.post("/benchmark-plan", response_model=BenchmarkPlanOut)
    def get_benchmark_plan(body: BenchmarkPlanRequest, db: Session = Depends(_get_db)):
        scenario = db.get(ScenarioRow, body.scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")

        from planner.benchmark_plan import benchmark_plan
        from planner.capacity import plan
        from planner.catalog import CatalogError, get_gpu, resolve_model

        try:
            model = resolve_model(scenario.model_name)
            gpu = get_gpu(scenario.gpu_name)
        except CatalogError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        try:
            est = plan(
                requests_per_day=scenario.requests_per_day,
                peak_multiplier=scenario.peak_multiplier,
                isl=scenario.isl,
                osl=scenario.osl,
                ttft_slo_ms=scenario.ttft_slo_ms,
                model=model,
                gpu=gpu,
                dtype=scenario.dtype,
                tp=scenario.tp,
                traffic_class=scenario.traffic_class,
                gpu_mem_util=scenario.gpu_mem_util,
            )
            bp = benchmark_plan(est, scenario.model_name, scenario.gpu_name)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        return BenchmarkPlanOut(
            steps=[
                BenchmarkStepOut(
                    priority=s.priority,
                    label=s.label,
                    command=s.command,
                    purpose=s.purpose,
                    collapses_confidence_on=s.collapses_confidence_on,
                )
                for s in bp.steps
            ],
            confidence=bp.confidence,
            binding_constraint=bp.binding_constraint,
            rationale=bp.rationale,
        )

    # ── POST /benchmarks/run ───────────────────────────────────────────────

    @app.post("/benchmarks/run", response_model=BenchmarkRunOut, status_code=202)
    def run_benchmark(
        body: RunBenchmarkRequest,
        request: Request,
        background_tasks: BackgroundTasks,
        db: Session = Depends(_get_db),
    ):
        scenario = db.get(ScenarioRow, body.scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")

        from planner.benchmark_plan import benchmark_plan
        from planner.capacity import plan
        from planner.catalog import CatalogError, get_gpu, resolve_model

        try:
            model = resolve_model(scenario.model_name)
            gpu = get_gpu(scenario.gpu_name)
        except CatalogError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        try:
            est = plan(
                requests_per_day=scenario.requests_per_day,
                peak_multiplier=scenario.peak_multiplier,
                isl=scenario.isl,
                osl=scenario.osl,
                ttft_slo_ms=scenario.ttft_slo_ms,
                model=model,
                gpu=gpu,
                dtype=scenario.dtype,
                tp=scenario.tp,
                traffic_class=scenario.traffic_class,
                gpu_mem_util=scenario.gpu_mem_util,
            )
            bp = benchmark_plan(est, scenario.model_name, scenario.gpu_name)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        if body.step_index >= len(bp.steps):
            raise HTTPException(
                status_code=422,
                detail=f"step_index {body.step_index} is out of range "
                       f"(plan has {len(bp.steps)} steps)",
            )

        step = bp.steps[body.step_index]

        # Extract tag from the generated command so result filename is deterministic
        parts = step.command.split()
        tag = parts[parts.index("--tag") + 1] if "--tag" in parts else f"scenario_{scenario.id}_s{body.step_index}"

        out_dir = Path(body.output_dir) if body.output_dir else request.app.state.results_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        result_path = str(out_dir / f"{tag}.json")

        # Strip basic-auth credentials from the URL so they don't sit in the
        # stored command in plaintext inside the URL string; pass them via
        # the dedicated --basic-auth flag instead.
        parsed = urlparse(body.endpoint)
        basic_auth_flag = ""
        if parsed.username or parsed.password:
            creds = f"{parsed.username or ''}:{parsed.password or ''}"
            clean = parsed._replace(netloc=parsed.hostname + (f":{parsed.port}" if parsed.port else ""))
            clean_endpoint = urlunparse(clean)
            basic_auth_flag = f" --basic-auth {creds}"
        else:
            clean_endpoint = body.endpoint

        full_command = f"{step.command} --endpoint {clean_endpoint} --output-dir {out_dir}{basic_auth_flag}"

        run_row = BenchmarkRunRow(
            scenario_id=scenario.id,
            tag=tag,
            command=full_command,
            status="queued",
            result_path=result_path,
            created_at=datetime.now(timezone.utc),
        )
        db.add(run_row)
        db.commit()
        db.refresh(run_row)

        background_tasks.add_task(
            run_job,
            run_id=run_row.id,
            command=full_command,
            result_path=result_path,
            session_factory=request.app.state.session_factory,
        )

        return run_row

    # ── GET /benchmarks/{id} ───────────────────────────────────────────────

    @app.get("/benchmarks/{run_id}", response_model=BenchmarkRunOut)
    def get_benchmark_run(run_id: int, db: Session = Depends(_get_db)):
        row = db.get(BenchmarkRunRow, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Benchmark run not found")
        return row

    # ── POST /ingest/{run_id} ──────────────────────────────────────────────

    @app.post("/ingest/{run_id}", response_model=IngestOut)
    def ingest_run(run_id: int, request: Request, db: Session = Depends(_get_db)):
        run = db.get(BenchmarkRunRow, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Benchmark run not found")
        if run.status != "done":
            raise HTTPException(
                status_code=422,
                detail=f"Run {run_id} has status '{run.status}'; expected 'done'",
            )
        if not run.result_path:
            raise HTTPException(status_code=422, detail="Run has no result_path recorded")

        result_path = Path(run.result_path)
        if not result_path.exists():
            raise HTTPException(
                status_code=422,
                detail=f"Result file not found: {run.result_path}",
            )

        scenario = db.get(ScenarioRow, run.scenario_id)

        from planner.catalog import CatalogError
        from planner.ingest_anchor import ingest_anchor

        anchors_file = request.app.state.anchors_file  # None = catalog default

        try:
            anchor, impact = ingest_anchor(
                result_path=result_path,
                gpu_name=scenario.gpu_name if scenario else "l4",
                dtype=scenario.dtype if scenario else "fp8",
                model_name=scenario.model_name if scenario else None,
                anchors_file=anchors_file,
            )
        except CatalogError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        return IngestOut(
            anchor={
                "model": anchor.model,
                "gpu": anchor.gpu,
                "dtype": anchor.dtype,
                "isl": anchor.isl,
                "osl": anchor.osl,
                "concurrency": anchor.concurrency,
                "measured_ttft_p50_ms": anchor.measured_ttft_p50_ms,
                "measured_throughput_tok_s": anchor.measured_throughput_tok_s,
                "derived_mfu_prefill": anchor.derived_mfu_prefill,
                "source": anchor.source,
            },
            impact=impact,
        )

    # ── GET /scenarios/{id}/recommendation ────────────────────────────────

    @app.get("/scenarios/{scenario_id}/recommendation", response_model=RecommendationOut)
    def get_recommendation(scenario_id: int, db: Session = Depends(_get_db)):
        scenario = db.get(ScenarioRow, scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")

        est_row = db.execute(
            select(CapacityEstimateRow)
            .where(CapacityEstimateRow.scenario_id == scenario_id)
            .order_by(desc(CapacityEstimateRow.created_at))
        ).scalars().first()

        if est_row is None:
            raise HTTPException(
                status_code=422,
                detail="No estimate found for this scenario. Call POST /estimate first.",
            )

        done_runs = db.execute(
            select(BenchmarkRunRow)
            .where(
                BenchmarkRunRow.scenario_id == scenario_id,
                BenchmarkRunRow.status == "done",
            )
        ).scalars().all()

        mode = "validated_by_benchmark" if done_runs else "estimate_only"

        summary = {
            "mode": mode,
            "replicas": est_row.replicas,
            "replicas_low": est_row.replicas_low,
            "replicas_high": est_row.replicas_high,
            "binding_constraint": est_row.binding_constraint,
            "confidence": est_row.confidence,
            "benchmark_runs_done": len(done_runs),
            "model": scenario.model_name,
            "gpu": scenario.gpu_name,
            "dtype": scenario.dtype,
        }

        # Upsert: update existing recommendation or create a new one
        existing = db.execute(
            select(RecommendationRow)
            .where(RecommendationRow.scenario_id == scenario_id)
            .order_by(desc(RecommendationRow.created_at))
        ).scalars().first()

        if existing:
            existing.summary = json.dumps(summary)
            existing.confidence = est_row.confidence
            db.commit()
            db.refresh(existing)
            rec_row = existing
        else:
            rec_row = RecommendationRow(
                scenario_id=scenario_id,
                summary=json.dumps(summary),
                confidence=est_row.confidence,
                created_at=datetime.now(timezone.utc),
            )
            db.add(rec_row)
            db.commit()
            db.refresh(rec_row)

        return RecommendationOut(
            id=rec_row.id,
            scenario_id=rec_row.scenario_id,
            summary=json.loads(rec_row.summary),
            confidence=rec_row.confidence,
            created_at=rec_row.created_at,
        )

    # ── GET /reports/{id} ─────────────────────────────────────────────────

    @app.get("/reports/{report_id}", response_model=ReportOut)
    def get_report(report_id: int, db: Session = Depends(_get_db)):
        row = db.get(ReportRow, report_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Report not found")
        return row

    return app


# Default instance for `uvicorn api.main:app`
# Render sets DATABASE_URL with postgres:// — SQLAlchemy requires postgresql://
_db_url = os.getenv("DATABASE_URL", "sqlite:///./capacity_planner.db")
if _db_url.startswith("postgres://"):
    _db_url = _db_url.replace("postgres://", "postgresql://", 1)

app = create_app(
    db_url=_db_url,
    results_dir=os.getenv("RESULTS_DIR", "results/real"),
)
