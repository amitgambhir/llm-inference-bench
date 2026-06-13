"""tests/test_api.py — Phase 4 API acceptance tests.

End-to-end flow tested:
  POST /scenarios → POST /estimate → POST /benchmark-plan
    → POST /benchmarks/run (mocked subprocess)
    → GET  /benchmarks/{id}
    → POST /ingest/{run_id}
    → GET  /scenarios/{id}/recommendation
"""
from __future__ import annotations

import json
import subprocess as _subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import planner.catalog as _catalog_module
from api.main import create_app


# ── Shared test data ──────────────────────────────────────────────────────────

SCENARIO_PAYLOAD = {
    "name": "test-l4-fp8",
    "model_name": "llama-3.1-8b",
    "gpu_name": "l4",
    "dtype": "fp8",
    "requests_per_day": 500_000,
    "peak_multiplier": 2.0,
    "isl": 512,
    "osl": 128,
    "ttft_slo_ms": 5000.0,
    "tp": 1,
    "traffic_class": "batch",
    "gpu_mem_util": 0.90,
    "runtime": "vllm",
}

# Minimal realistic result that ingest_anchor can parse
FAKE_RESULT = {
    "meta": {
        "tag": "api_test_run",
        "model": "llama-3.1-8b",
        "gpu": {"name": "NVIDIA L4", "memory_mb": 23034},
        "workload": {
            "isl_approx": 512,
            "osl_max": 128,
            "concurrency": 10,
            "duration_secs": 60,
        },
        "synthetic": False,
    },
    "metrics": {
        "ttft_ms": {"p50": 75.0, "p95": 81.0},
        "throughput_tokens_per_sec": 262.0,
        "throughput_req_per_sec": 2.05,
    },
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_catalog():
    _catalog_module._catalog = None
    yield
    _catalog_module._catalog = None


@pytest.fixture
def test_app(tmp_path):
    """Isolated app instance — fresh SQLite DB + temp results/anchors dirs."""
    results_dir = tmp_path / "results" / "real"
    results_dir.mkdir(parents=True, exist_ok=True)
    anchors_file = tmp_path / "test_anchors.yaml"
    return create_app(
        db_url=f"sqlite:///{tmp_path}/test.db",
        results_dir=str(results_dir),
        anchors_file=str(anchors_file),
    )


@pytest.fixture
def client(test_app):
    with TestClient(test_app) as c:
        yield c


@pytest.fixture
def scenario_id(client):
    r = client.post("/scenarios", json=SCENARIO_PAYLOAD)
    assert r.status_code == 201
    return r.json()["id"]


@pytest.fixture
def estimate_id(client, scenario_id):
    r = client.post("/estimate", json={"scenario_id": scenario_id})
    assert r.status_code == 201
    return r.json()["id"]


@pytest.fixture
def mock_subprocess(monkeypatch, test_app):
    """Patch subprocess.run so it writes a fake result file and exits 0."""

    def fake_run(cmd, **kwargs):
        parts = cmd if isinstance(cmd, list) else cmd.split()
        tag = parts[parts.index("--tag") + 1] if "--tag" in parts else "unknown"
        try:
            out_dir = Path(parts[parts.index("--output-dir") + 1])
        except (ValueError, IndexError):
            out_dir = test_app.state.results_dir

        out_dir.mkdir(parents=True, exist_ok=True)
        result = json.loads(json.dumps(FAKE_RESULT))   # deep copy
        result["meta"]["tag"] = tag
        (out_dir / f"{tag}.json").write_text(json.dumps(result))
        return _subprocess.CompletedProcess(parts, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(_subprocess, "run", fake_run)


@pytest.fixture
def done_run(client, scenario_id, mock_subprocess):
    """A benchmark run that has completed (background task ran synchronously)."""
    r = client.post(
        "/benchmarks/run",
        json={"scenario_id": scenario_id, "endpoint": "http://localhost:8000"},
    )
    assert r.status_code == 202
    return r.json()


# ── POST /scenarios ───────────────────────────────────────────────────────────


def test_create_scenario_201(client):
    r = client.post("/scenarios", json=SCENARIO_PAYLOAD)
    assert r.status_code == 201


def test_create_scenario_has_id(client):
    data = client.post("/scenarios", json=SCENARIO_PAYLOAD).json()
    assert isinstance(data["id"], int) and data["id"] >= 1


def test_create_scenario_stores_fields(client):
    data = client.post("/scenarios", json=SCENARIO_PAYLOAD).json()
    assert data["model_name"] == "llama-3.1-8b"
    assert data["gpu_name"] == "l4"
    assert data["isl"] == 512
    assert data["osl"] == 128


# ── POST /estimate ─────────────────────────────────────────────────────────────


def test_estimate_201(client, scenario_id):
    r = client.post("/estimate", json={"scenario_id": scenario_id})
    assert r.status_code == 201


def test_estimate_replicas_positive(client, scenario_id):
    data = client.post("/estimate", json={"scenario_id": scenario_id}).json()
    assert data["replicas"] >= 1


def test_estimate_range_ordered(client, scenario_id):
    data = client.post("/estimate", json={"scenario_id": scenario_id}).json()
    assert data["replicas_low"] <= data["replicas"] <= data["replicas_high"]


def test_estimate_confidence_valid(client, scenario_id):
    data = client.post("/estimate", json={"scenario_id": scenario_id}).json()
    assert data["confidence"] in ("high", "medium", "low")


def test_estimate_binding_constraint_valid(client, scenario_id):
    data = client.post("/estimate", json={"scenario_id": scenario_id}).json()
    assert data["binding_constraint"] in ("prefill-bound", "decode-bound", "kv-memory-bound")


def test_estimate_payload_has_warnings(client, scenario_id):
    data = client.post("/estimate", json={"scenario_id": scenario_id}).json()
    assert "payload" in data
    assert "warnings" in data["payload"]


def test_estimate_unknown_scenario_404(client):
    r = client.post("/estimate", json={"scenario_id": 9999})
    assert r.status_code == 404


# ── POST /benchmark-plan ──────────────────────────────────────────────────────


def test_benchmark_plan_returns_steps(client, scenario_id):
    data = client.post("/benchmark-plan", json={"scenario_id": scenario_id}).json()
    assert len(data["steps"]) >= 5


def test_benchmark_plan_commands_valid(client, scenario_id):
    data = client.post("/benchmark-plan", json={"scenario_id": scenario_id}).json()
    for step in data["steps"]:
        assert step["command"].startswith("python collect/run_bench.py")
        assert "--isl" in step["command"]
        assert "--tag" in step["command"]


def test_benchmark_plan_has_rationale(client, scenario_id):
    data = client.post("/benchmark-plan", json={"scenario_id": scenario_id}).json()
    assert len(data["rationale"]) > 10


def test_benchmark_plan_unknown_scenario_404(client):
    r = client.post("/benchmark-plan", json={"scenario_id": 9999})
    assert r.status_code == 404


# ── POST /benchmarks/run ──────────────────────────────────────────────────────


def test_run_benchmark_202(client, scenario_id, mock_subprocess):
    r = client.post(
        "/benchmarks/run",
        json={"scenario_id": scenario_id, "endpoint": "http://gpu:8000"},
    )
    assert r.status_code == 202


def test_run_benchmark_has_run_id(client, scenario_id, mock_subprocess):
    data = client.post(
        "/benchmarks/run",
        json={"scenario_id": scenario_id, "endpoint": "http://gpu:8000"},
    ).json()
    assert isinstance(data["id"], int) and data["id"] >= 1


def test_run_benchmark_command_includes_endpoint(client, scenario_id, mock_subprocess):
    data = client.post(
        "/benchmarks/run",
        json={"scenario_id": scenario_id, "endpoint": "http://gpu-server:8001"},
    ).json()
    assert "http://gpu-server:8001" in data["command"]


def test_run_benchmark_background_task_completes(client, scenario_id, mock_subprocess):
    """BackgroundTasks run synchronously inside Starlette TestClient."""
    run_id = client.post(
        "/benchmarks/run",
        json={"scenario_id": scenario_id, "endpoint": "http://localhost:8000"},
    ).json()["id"]

    status = client.get(f"/benchmarks/{run_id}").json()
    assert status["status"] == "done"


def test_run_benchmark_invalid_step_index_422(client, scenario_id, mock_subprocess):
    r = client.post(
        "/benchmarks/run",
        json={"scenario_id": scenario_id, "endpoint": "http://localhost:8000", "step_index": 9999},
    )
    assert r.status_code == 422


def test_run_benchmark_unknown_scenario_404(client, mock_subprocess):
    r = client.post(
        "/benchmarks/run",
        json={"scenario_id": 9999, "endpoint": "http://localhost:8000"},
    )
    assert r.status_code == 404


# ── GET /benchmarks/{id} ──────────────────────────────────────────────────────


def test_get_benchmark_run_not_found(client):
    assert client.get("/benchmarks/9999").status_code == 404


def test_get_benchmark_run_returns_fields(client, done_run):
    data = client.get(f"/benchmarks/{done_run['id']}").json()
    assert data["tag"]
    assert data["command"]
    assert data["status"] == "done"
    assert data["result_path"]


# ── POST /ingest/{run_id} ─────────────────────────────────────────────────────


def test_ingest_returns_anchor(client, done_run):
    data = client.post(f"/ingest/{done_run['id']}").json()
    assert data["anchor"]["model"] == "llama-3.1-8b"
    assert data["anchor"]["gpu"] == "l4"
    assert data["anchor"]["isl"] == 512


def test_ingest_returns_impact_lines(client, done_run):
    data = client.post(f"/ingest/{done_run['id']}").json()
    assert isinstance(data["impact"], list)
    assert len(data["impact"]) >= 1


def test_ingest_derived_mfu_in_range(client, done_run):
    data = client.post(f"/ingest/{done_run['id']}").json()
    mfu = data["anchor"]["derived_mfu_prefill"]
    assert 0.01 <= mfu <= 0.99


def test_ingest_queued_run_returns_422(client, scenario_id):
    """Cannot ingest a run that has not finished."""
    from api.db import BenchmarkRunRow

    session = client.app.state.session_factory()
    try:
        run = BenchmarkRunRow(
            scenario_id=scenario_id,
            tag="queued_tag",
            command="python collect/run_bench.py --tag queued_tag",
            status="queued",
            created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )
        session.add(run)
        session.commit()
        run_id = run.id
    finally:
        session.close()

    assert client.post(f"/ingest/{run_id}").status_code == 422


def test_ingest_unknown_run_404(client):
    assert client.post("/ingest/9999").status_code == 404


# ── GET /scenarios/{id}/recommendation ───────────────────────────────────────


def test_recommendation_requires_estimate(client, scenario_id):
    assert client.get(f"/scenarios/{scenario_id}/recommendation").status_code == 422


def test_recommendation_returns_summary(client, scenario_id, estimate_id):
    data = client.get(f"/scenarios/{scenario_id}/recommendation").json()
    assert "summary" in data
    assert data["summary"]["replicas"] >= 1


def test_recommendation_mode_estimate_only(client, scenario_id, estimate_id):
    data = client.get(f"/scenarios/{scenario_id}/recommendation").json()
    assert data["summary"]["mode"] == "estimate_only"


def test_recommendation_confidence_valid(client, scenario_id, estimate_id):
    data = client.get(f"/scenarios/{scenario_id}/recommendation").json()
    assert data["confidence"] in ("high", "medium", "low")


def test_recommendation_unknown_scenario_404(client):
    assert client.get("/scenarios/9999/recommendation").status_code == 404


# ── GET /reports/{id} ────────────────────────────────────────────────────────


def test_get_report_not_found(client):
    assert client.get("/reports/9999").status_code == 404


# ── End-to-end acceptance flow ────────────────────────────────────────────────


def test_full_planning_workflow(client, mock_subprocess):
    """Acceptance: scenario → estimate → plan → run → ingest → recommendation."""

    # 1. Create scenario
    scenario = client.post("/scenarios", json=SCENARIO_PAYLOAD).json()
    assert scenario["id"] >= 1
    sid = scenario["id"]

    # 2. Run estimate
    est = client.post("/estimate", json={"scenario_id": sid}).json()
    assert est["replicas"] >= 1
    assert est["confidence"] in ("high", "medium", "low")
    assert "warnings" in est["payload"]

    # 3. Benchmark plan
    plan = client.post("/benchmark-plan", json={"scenario_id": sid}).json()
    assert len(plan["steps"]) >= 5
    assert all(s["command"].startswith("python collect/run_bench.py") for s in plan["steps"])

    # 4. Run benchmark (subprocess is mocked → completes synchronously)
    run = client.post(
        "/benchmarks/run",
        json={"scenario_id": sid, "endpoint": "http://gpu-server:8000"},
    ).json()
    assert run["scenario_id"] == sid
    run_id = run["id"]

    # 5. Poll status — background task completes before TestClient returns
    status = client.get(f"/benchmarks/{run_id}").json()
    assert status["status"] == "done"
    assert status["exit_code"] == 0

    # 6. Ingest anchor from completed result
    ingest = client.post(f"/ingest/{run_id}").json()
    assert ingest["anchor"]["model"] == "llama-3.1-8b"
    assert len(ingest["impact"]) >= 1

    # 7. Recommendation reflects the validated benchmark
    rec = client.get(f"/scenarios/{sid}/recommendation").json()
    assert rec["summary"]["mode"] == "validated_by_benchmark"
    assert rec["summary"]["benchmark_runs_done"] == 1
    assert rec["confidence"] in ("high", "medium", "low")
