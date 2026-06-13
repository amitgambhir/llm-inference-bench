"""Phase 5 tests for planner/report.py."""
import pytest
from datetime import datetime, timezone

from planner.catalog import get_gpu, get_model
from planner.capacity import plan
from planner.cost import compute_cost
from planner.report import render_report
import planner.catalog as _catalog_module


@pytest.fixture(autouse=True)
def reset_catalog():
    _catalog_module._catalog = None
    yield
    _catalog_module._catalog = None


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def golden_estimate():
    """gpt-oss-20b / H100 / mxfp4 — the plan's canonical golden example."""
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    return plan(
        requests_per_day=100_000_000,
        peak_multiplier=3.0,
        isl=9000, osl=500,
        ttft_slo_ms=2000.0,
        model=model, gpu=gpu, dtype="mxfp4",
        tp=1, traffic_class="realtime",
    )


@pytest.fixture
def high_conf_estimate():
    """llama-3.1-8b / L4 / fp8 — HIGH confidence, seeded anchors."""
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    return plan(
        requests_per_day=500_000,
        peak_multiplier=2.0,
        isl=2048, osl=128,
        ttft_slo_ms=5000.0,
        model=model, gpu=gpu, dtype="fp8",
        tp=1, traffic_class="batch",
    )


def _base_report(estimate, mode="estimate_only", cost=None, anchors=None):
    return render_report(
        estimate=estimate,
        scenario_label="test-scenario",
        model_name="gpt-oss-20b",
        gpu_name="h100_sxm",
        dtype="mxfp4",
        isl=9000, osl=500,
        requests_per_day=100_000_000,
        peak_multiplier=3.0,
        tp=1, traffic_class="realtime",
        cost=cost,
        anchors_used=anchors,
        mode=mode,
        generated_at=datetime(2026, 6, 13, 10, 0, 0, tzinfo=timezone.utc),
    )


# ── Return type and basic structure ──────────────────────────────────────────


def test_render_report_returns_string(golden_estimate):
    result = _base_report(golden_estimate)
    assert isinstance(result, str)


def test_render_report_non_empty(golden_estimate):
    result = _base_report(golden_estimate)
    assert len(result) > 200


def test_render_report_is_markdown(golden_estimate):
    result = _base_report(golden_estimate)
    assert result.startswith("# ")


# ── Mode badge ────────────────────────────────────────────────────────────────


def test_estimate_only_mode_badge(golden_estimate):
    result = _base_report(golden_estimate, mode="estimate_only")
    assert "ESTIMATE ONLY" in result


def test_partially_validated_mode_badge(golden_estimate):
    result = _base_report(golden_estimate, mode="partially_validated")
    assert "PARTIALLY VALIDATED" in result


def test_validated_mode_badge(golden_estimate):
    result = _base_report(golden_estimate, mode="validated_by_benchmark")
    assert "VALIDATED BY BENCHMARK" in result


# ── Confidence ────────────────────────────────────────────────────────────────


def test_report_includes_confidence_label_low(golden_estimate):
    result = _base_report(golden_estimate)
    assert "LOW" in result


def test_report_includes_confidence_label_high(high_conf_estimate):
    result = render_report(
        estimate=high_conf_estimate,
        scenario_label="high-conf",
        model_name="llama-3.1-8b",
        gpu_name="l4",
        dtype="fp8",
        isl=2048, osl=128,
        requests_per_day=500_000,
        peak_multiplier=2.0,
    )
    assert "HIGH" in result


def test_report_includes_band_percentage(golden_estimate):
    result = _base_report(golden_estimate)
    assert "±50%" in result   # LOW confidence = ±50%


# ── Sizing figures ────────────────────────────────────────────────────────────


def test_report_includes_replicas(golden_estimate):
    result = _base_report(golden_estimate)
    assert str(golden_estimate.replicas) in result


def test_report_includes_replicas_range(golden_estimate):
    result = _base_report(golden_estimate)
    assert str(golden_estimate.replicas_low) in result
    assert str(golden_estimate.replicas_high) in result


def test_report_includes_binding_constraint(golden_estimate):
    result = _base_report(golden_estimate)
    assert golden_estimate.binding_constraint in result


def test_report_includes_ttft(golden_estimate):
    result = _base_report(golden_estimate)
    assert "ms" in result.lower()
    assert "TTFT" in result or "ttft" in result.lower()


# ── Workload section ──────────────────────────────────────────────────────────


def test_report_includes_model_name(golden_estimate):
    result = _base_report(golden_estimate)
    assert "gpt-oss-20b" in result


def test_report_includes_gpu_name(golden_estimate):
    result = _base_report(golden_estimate)
    assert "h100_sxm" in result


def test_report_includes_isl_osl(golden_estimate):
    result = _base_report(golden_estimate)
    assert "9000" in result
    assert "500" in result


# ── Warnings ─────────────────────────────────────────────────────────────────


def test_report_includes_warnings_section_when_warnings_exist(golden_estimate):
    assert golden_estimate.warnings   # golden estimate always has warnings
    result = _base_report(golden_estimate)
    assert "Warning" in result or "⚠" in result


# ── Cost section ─────────────────────────────────────────────────────────────


def test_report_includes_cost_section_when_provided(golden_estimate):
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    result = _base_report(golden_estimate, cost=cost)
    assert "Cost Envelope" in result
    assert "$" in result


def test_report_omits_cost_section_when_not_provided(golden_estimate):
    result = _base_report(golden_estimate, cost=None)
    assert "Cost Envelope" not in result


# ── Next steps ────────────────────────────────────────────────────────────────


def test_estimate_only_next_steps_mention_benchmark(golden_estimate):
    result = _base_report(golden_estimate, mode="estimate_only")
    assert "benchmark" in result.lower()


def test_validated_next_steps_mention_share(golden_estimate):
    result = _base_report(golden_estimate, mode="validated_by_benchmark")
    assert "share" in result.lower()


# ── Anchor evidence section ───────────────────────────────────────────────────


def test_estimate_only_no_anchor_evidence(golden_estimate):
    result = _base_report(golden_estimate, mode="estimate_only")
    assert "No benchmark data available" in result


def test_validated_with_anchors_lists_them(golden_estimate):
    anchors = ["l4/fp8/llama-3.1-8b ISL=2048 c=10 (ingest from run_001)"]
    result = _base_report(
        golden_estimate,
        mode="validated_by_benchmark",
        anchors=anchors,
    )
    assert "run_001" in result


# ── Timestamp ─────────────────────────────────────────────────────────────────


def test_report_includes_timestamp(golden_estimate):
    result = _base_report(golden_estimate)
    assert "2026-06-13" in result
