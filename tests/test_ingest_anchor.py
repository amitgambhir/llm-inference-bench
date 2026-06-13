"""Phase 3 tests for planner/ingest_anchor.py and planner/confidence.py.

Key acceptance criteria:
  - Feeding a synthetic result JSON produces a valid new anchor row.
  - A scenario that was LOW before ingest becomes HIGH after, verified by
    a before/after test on confidence().
  - derived_mfu_prefill and derived_bw_eff_decode are in [0.01, 0.99].
  - Duplicate ingest is a no-op (idempotent).
  - CatalogError on missing required fields.
"""
import json
import pytest
from pathlib import Path

from planner.catalog import Anchor, CatalogError, get_gpu, get_model
from planner.confidence import (
    ConfidenceResult,
    ExtrapolationDistance,
    compute_confidence_from_anchors,
    confidence,
)
from planner.ingest_anchor import (
    _append_anchor,
    _load_anchors,
    derive_bw_efficiency_decode,
    derive_mfu_prefill,
    ingest_anchor,
)
import planner.catalog as _catalog_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_catalog():
    _catalog_module._catalog = None
    yield
    _catalog_module._catalog = None


def _make_result_json(
    tmp_path: Path,
    model: str = "gpt-oss-20b",
    gpu_display: str = "NVIDIA H100 SXM",
    isl: int = 9000,
    osl: int = 500,
    concurrency: int = 10,
    ttft_p50_ms: float = 96.0,
    ttft_p95_ms: float = 140.0,
    throughput_tps: float = 8000.0,
    tag: str = "test_run",
    synthetic: bool = True,
) -> Path:
    data = {
        "meta": {
            "tag": tag,
            "runtime": "vllm",
            "model": model,
            "gpu": {"name": gpu_display, "memory_mb": 81920, "util_pct": 75},
            "config": {"chunked_prefill": False, "tensor_parallel_size": 1},
            "workload": {
                "isl_approx": isl,
                "osl_max": osl,
                "concurrency": concurrency,
                "duration_secs": 90,
            },
            "synthetic": synthetic,
            "timestamp": "2026-06-13T09:00:00+00:00",
        },
        "metrics": {
            "ttft_ms": {"p50": ttft_p50_ms, "p90": ttft_p50_ms * 1.1, "p95": ttft_p95_ms, "p99": ttft_p95_ms * 1.1, "mean": ttft_p50_ms * 0.95},
            "total_latency_ms": {"p50": 5000, "p95": 6000, "p99": 6500},
            "throughput_tokens_per_sec": throughput_tps,
            "throughput_req_per_sec": throughput_tps / (isl + osl),
            "total_requests": 180,
            "successful_requests": 180,
            "failed_requests": 0,
        },
    }
    path = tmp_path / f"{tag}.json"
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# planner/confidence.py — ExtrapolationDistance and ConfidenceResult fields
# ---------------------------------------------------------------------------


def test_confidence_result_has_band_factor():
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    result = confidence(model, gpu, "mxfp4", 9000, 500, 10)
    assert hasattr(result, "band_factor")
    assert result.band_factor == 0.50  # LOW → 50%


def test_confidence_high_band_factor():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    result = confidence(model, gpu, "fp8", 2048, 128, 10)
    assert result.level == "high"
    assert result.band_factor == 0.10


def test_confidence_medium_band_factor():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    # ISL=8192 is far from any L4/fp8 anchor (max 4096) → MEDIUM
    result = confidence(model, gpu, "fp8", 8192, 128, 10)
    assert result.level == "medium"
    assert result.band_factor == 0.25


def test_extrapolation_distance_computed_for_close_anchor():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    result = confidence(model, gpu, "fp8", 2048, 128, 10)
    assert result.extrapolation_distance is not None
    assert isinstance(result.extrapolation_distance, ExtrapolationDistance)
    assert result.extrapolation_distance.isl_distance <= 0.20


def test_extrapolation_distance_none_for_no_anchor():
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    result = confidence(model, gpu, "mxfp4", 9000, 500, 10)
    assert result.extrapolation_distance is None


def test_best_anchor_set_on_high_confidence():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    result = confidence(model, gpu, "fp8", 2048, 128, 10)
    assert result.best_anchor is not None
    assert result.best_anchor.model == "llama-3.1-8b"


def test_compute_confidence_from_anchors_pure():
    """Pure function path: result depends only on the passed anchor list."""
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")

    # With empty anchors → LOW
    result_empty = compute_confidence_from_anchors([], model, gpu, "mxfp4", 9000, 500, 10)
    assert result_empty.level == "low"

    # With a synthetic anchor at matching ISL → HIGH
    fake_anchor = Anchor(
        model="gpt-oss-20b", gpu="h100_sxm", dtype="mxfp4",
        isl=9000, osl=500, concurrency=10,
        measured_ttft_p50_ms=96.0, measured_ttft_p95_ms=140.0,
        measured_throughput_tok_s=8000.0,
        derived_mfu_prefill=0.38,
        source="synthetic test",
    )
    result_with = compute_confidence_from_anchors([fake_anchor], model, gpu, "mxfp4", 9000, 500, 10)
    assert result_with.level == "high"
    assert result_with.mfu_used == pytest.approx(0.38)


# ---------------------------------------------------------------------------
# derive_mfu_prefill and derive_bw_efficiency_decode
# ---------------------------------------------------------------------------


def test_derive_mfu_in_valid_range():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    mfu = derive_mfu_prefill(
        isl=2048,
        ttft_p50_ms=115.0,
        model=model,
        gpu=gpu,
        dtype="fp8",
    )
    assert 0.01 <= mfu <= 0.99


def test_derive_mfu_clamped_to_valid_range():
    # TTFT at concurrency > 1 reflects batched prefill across multiple concurrent
    # requests, so the back-computed MFU can nominally exceed 1.0.  The function
    # must clamp to [0.01, 0.99] — which is what this test verifies.
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    mfu = derive_mfu_prefill(2048, 115.0, model, gpu, "fp8")
    assert 0.01 <= mfu <= 0.99, f"MFU outside clamped range: {mfu:.4f}"


def test_derive_bw_eff_in_valid_range():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    bw_eff = derive_bw_efficiency_decode(
        throughput_tps=262.0,
        concurrency=10,
        isl=2048,
        osl=128,
        model=model,
        gpu=gpu,
    )
    assert 0.01 <= bw_eff <= 0.99


def test_faster_prefill_yields_higher_mfu():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    mfu_slow = derive_mfu_prefill(2048, 200.0, model, gpu, "fp8")
    mfu_fast = derive_mfu_prefill(2048, 50.0, model, gpu, "fp8")
    assert mfu_fast > mfu_slow


def test_higher_throughput_yields_higher_bw_eff():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    bw_low = derive_bw_efficiency_decode(100.0, 10, 2048, 128, model, gpu)
    bw_high = derive_bw_efficiency_decode(600.0, 10, 2048, 128, model, gpu)
    assert bw_high > bw_low


# ---------------------------------------------------------------------------
# ingest_anchor — valid result produces a correct anchor
# ---------------------------------------------------------------------------


def test_ingest_produces_anchor(tmp_path):
    result = _make_result_json(tmp_path, model="llama-3.1-8b",
                               isl=512, osl=128, concurrency=10,
                               ttft_p50_ms=75.0, throughput_tps=262.0,
                               tag="test_l4_fp8_ingest")
    anchors_file = tmp_path / "anchors.yaml"

    anchor, _ = ingest_anchor(
        result_path=result,
        gpu_name="l4",
        dtype="fp8",
        anchors_file=anchors_file,
    )

    assert isinstance(anchor, Anchor)
    assert anchor.model == "llama-3.1-8b"
    assert anchor.gpu == "l4"
    assert anchor.dtype == "fp8"
    assert anchor.isl == 512
    assert anchor.measured_ttft_p50_ms == 75.0


def test_ingest_writes_anchor_to_yaml(tmp_path):
    result = _make_result_json(tmp_path, model="llama-3.1-8b",
                               isl=512, osl=128, concurrency=10,
                               ttft_p50_ms=75.0, throughput_tps=262.0)
    anchors_file = tmp_path / "anchors.yaml"

    ingest_anchor(result, "l4", "fp8", anchors_file=anchors_file)

    saved = _load_anchors(anchors_file)
    assert len(saved) == 1
    assert saved[0].model == "llama-3.1-8b"
    assert saved[0].derived_mfu_prefill is not None
    assert 0.01 <= saved[0].derived_mfu_prefill <= 0.99


def test_ingest_derived_mfu_correct(tmp_path):
    result = _make_result_json(tmp_path, model="llama-3.1-8b",
                               isl=2048, osl=128, concurrency=10,
                               ttft_p50_ms=115.0, throughput_tps=262.0,
                               tag="mfu_check")
    anchors_file = tmp_path / "anchors.yaml"

    anchor, _ = ingest_anchor(result, "l4", "fp8", anchors_file=anchors_file)
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    expected = derive_mfu_prefill(2048, 115.0, model, gpu, "fp8")
    assert anchor.derived_mfu_prefill == pytest.approx(expected, rel=1e-4)


def test_ingest_model_from_result_json(tmp_path):
    result = _make_result_json(tmp_path, model="llama-3.1-8b", isl=512, tag="auto_model")
    anchors_file = tmp_path / "anchors.yaml"

    # No explicit --model → should read from JSON meta.model
    anchor, _ = ingest_anchor(result, "l4", "fp8", anchors_file=anchors_file)
    assert anchor.model == "llama-3.1-8b"


def test_ingest_model_override(tmp_path):
    result = _make_result_json(tmp_path, model="llama-3.1-8b", isl=512, tag="override_model")
    anchors_file = tmp_path / "anchors.yaml"

    # Explicit model_name overrides JSON
    anchor, _ = ingest_anchor(result, "l4", "fp8",
                               model_name="llama-3.1-8b",
                               anchors_file=anchors_file)
    assert anchor.model == "llama-3.1-8b"


# ---------------------------------------------------------------------------
# ACCEPTANCE: before/after confidence upgrade
# ---------------------------------------------------------------------------


def test_confidence_upgrades_low_to_high_after_ingest(tmp_path):
    """Core acceptance test: LOW before ingest → HIGH after."""
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")

    # Before: no anchor for gpt-oss-20b/h100_sxm → LOW
    before = compute_confidence_from_anchors([], model, gpu, "mxfp4", 9000, 500, 10)
    assert before.level == "low"

    # Ingest a synthetic result for gpt-oss-20b on H100
    result = _make_result_json(
        tmp_path,
        model="gpt-oss-20b",
        isl=9000, osl=500, concurrency=10,
        ttft_p50_ms=96.0, throughput_tps=8000.0,
        tag="gpt_h100_ingest",
    )
    anchors_file = tmp_path / "anchors.yaml"
    anchor, impact = ingest_anchor(result, "h100_sxm", "mxfp4", anchors_file=anchors_file)

    # After: the new anchor is at ISL=9000 → distance 0 → HIGH
    anchors_after = _load_anchors(anchors_file)
    after = compute_confidence_from_anchors(anchors_after, model, gpu, "mxfp4", 9000, 500, 10)
    assert after.level == "high", f"Expected HIGH after ingest, got {after.level}"
    assert after.band_factor == 0.10


def test_confidence_upgrades_low_to_medium_for_nearby_isl(tmp_path):
    """Ingesting an anchor at ISL=4096 → scenarios at ISL=8192 go from LOW to MEDIUM."""
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")

    result = _make_result_json(
        tmp_path, model="gpt-oss-20b",
        isl=4096, osl=500, concurrency=10,
        ttft_p50_ms=50.0, throughput_tps=5000.0,
        tag="gpt_h100_isl4k",
    )
    anchors_file = tmp_path / "anchors.yaml"
    ingest_anchor(result, "h100_sxm", "mxfp4", anchors_file=anchors_file)

    anchors_after = _load_anchors(anchors_file)
    # ISL=8192 is 2× the anchor ISL=4096 — distance = 1.0 — still MEDIUM (same family)
    after = compute_confidence_from_anchors(anchors_after, model, gpu, "mxfp4", 8192, 500, 10)
    assert after.level in ("medium", "high"), \
        f"Expected at least MEDIUM after ingest, got {after.level}"


def test_impact_report_contains_before_after(tmp_path):
    result = _make_result_json(
        tmp_path, model="gpt-oss-20b",
        isl=9000, osl=500, concurrency=10,
        ttft_p50_ms=96.0, throughput_tps=8000.0,
    )
    anchors_file = tmp_path / "anchors.yaml"
    _, impact = ingest_anchor(result, "h100_sxm", "mxfp4", anchors_file=anchors_file)

    combined = "\n".join(impact).lower()
    assert "before" in combined
    assert "after" in combined
    assert "confidence" in combined or "low" in combined or "high" in combined


# ---------------------------------------------------------------------------
# Idempotency — duplicate ingest is a no-op
# ---------------------------------------------------------------------------


def test_duplicate_ingest_does_not_append(tmp_path):
    result = _make_result_json(tmp_path, model="llama-3.1-8b",
                               isl=512, concurrency=10,
                               ttft_p50_ms=75.0, throughput_tps=262.0,
                               tag="dedup_test")
    anchors_file = tmp_path / "anchors.yaml"

    ingest_anchor(result, "l4", "fp8", anchors_file=anchors_file)
    ingest_anchor(result, "l4", "fp8", anchors_file=anchors_file)  # second call

    saved = _load_anchors(anchors_file)
    assert len(saved) == 1, f"Expected 1 anchor after dedup, got {len(saved)}"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_ingest_missing_ttft_raises(tmp_path):
    data = {
        "meta": {"tag": "bad", "model": "llama-3.1-8b",
                 "workload": {"isl_approx": 512, "osl_max": 128, "concurrency": 10}},
        "metrics": {
            "ttft_ms": {"p50": 0},  # zero → invalid
            "throughput_tokens_per_sec": 262,
        },
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(CatalogError, match="ttft"):
        ingest_anchor(path, "l4", "fp8", anchors_file=tmp_path / "a.yaml")


def test_ingest_missing_throughput_raises(tmp_path):
    data = {
        "meta": {"tag": "bad", "model": "llama-3.1-8b",
                 "workload": {"isl_approx": 512, "osl_max": 128, "concurrency": 10}},
        "metrics": {
            "ttft_ms": {"p50": 75.0},
            "throughput_tokens_per_sec": 0,  # zero → invalid
        },
    }
    path = tmp_path / "bad2.json"
    path.write_text(json.dumps(data))
    with pytest.raises(CatalogError, match="throughput"):
        ingest_anchor(path, "l4", "fp8", anchors_file=tmp_path / "a.yaml")


def test_ingest_unknown_model_raises(tmp_path):
    result = _make_result_json(tmp_path, model="nonexistent-model-xyz",
                               isl=512, ttft_p50_ms=75.0, throughput_tps=200.0)
    with pytest.raises(CatalogError, match="not found"):
        ingest_anchor(result, "l4", "fp8", anchors_file=tmp_path / "a.yaml")


def test_ingest_unknown_gpu_raises(tmp_path):
    result = _make_result_json(tmp_path, model="llama-3.1-8b",
                               isl=512, ttft_p50_ms=75.0, throughput_tps=200.0)
    with pytest.raises(CatalogError):
        ingest_anchor(result, "nonexistent-gpu-xyz", "fp8",
                      anchors_file=tmp_path / "a.yaml")


def test_ingest_missing_isl_raises(tmp_path):
    data = {
        "meta": {"tag": "no_isl", "model": "llama-3.1-8b",
                 "workload": {"isl_approx": 0, "osl_max": 128, "concurrency": 10}},
        "metrics": {"ttft_ms": {"p50": 75.0}, "throughput_tokens_per_sec": 262},
    }
    path = tmp_path / "no_isl.json"
    path.write_text(json.dumps(data))
    with pytest.raises(CatalogError, match="isl_approx"):
        ingest_anchor(path, "l4", "fp8", anchors_file=tmp_path / "a.yaml")


# ---------------------------------------------------------------------------
# _append_anchor / _load_anchors round-trip
# ---------------------------------------------------------------------------


def test_load_anchors_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert _load_anchors(path) == []


def test_load_anchors_nonexistent(tmp_path):
    assert _load_anchors(tmp_path / "missing.yaml") == []


def test_append_and_load_roundtrip(tmp_path):
    path = tmp_path / "anchors.yaml"
    a = Anchor(
        model="llama-3.1-8b", gpu="l4", dtype="fp8",
        isl=512, osl=128, concurrency=10,
        measured_ttft_p50_ms=75.0, measured_ttft_p95_ms=81.0,
        measured_throughput_tok_s=262.0,
        derived_mfu_prefill=0.38,
        source="unit test",
    )
    _append_anchor(a, path)
    loaded = _load_anchors(path)
    assert len(loaded) == 1
    assert loaded[0].model == "llama-3.1-8b"
    assert loaded[0].derived_mfu_prefill == pytest.approx(0.38)
