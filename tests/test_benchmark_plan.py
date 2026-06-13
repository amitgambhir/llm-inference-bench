"""Phase 2 tests for planner/benchmark_plan.py."""
import pytest

from planner.catalog import get_gpu, get_model
from planner.capacity import plan
from planner.benchmark_plan import (
    BenchmarkPlan,
    BenchmarkStep,
    benchmark_plan,
    _infer_isl,
    _infer_osl,
)
import planner.catalog as _catalog_module


@pytest.fixture(autouse=True)
def reset_catalog():
    _catalog_module._catalog = None
    yield
    _catalog_module._catalog = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def golden_estimate():
    """LOW confidence, prefill-bound (gpt-oss-20b / H100 / mxfp4)."""
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    return plan(
        requests_per_day=100_000_000,
        peak_multiplier=3.0,
        isl=9000,
        osl=500,
        ttft_slo_ms=2000.0,
        model=model,
        gpu=gpu,
        dtype="mxfp4",
        tp=1,
        traffic_class="realtime",
    )


@pytest.fixture
def high_conf_estimate():
    """HIGH confidence (llama-3.1-8b / L4 / fp8 — backed by seeded anchors)."""
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    return plan(
        requests_per_day=500_000,
        peak_multiplier=2.0,
        isl=2048,
        osl=128,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="fp8",
        tp=1,
        traffic_class="batch",
    )


# ---------------------------------------------------------------------------
# Structure: return type, required fields
# ---------------------------------------------------------------------------


def test_benchmark_plan_returns_plan_object(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    assert isinstance(bp, BenchmarkPlan)


def test_each_step_has_required_fields(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    for step in bp.steps:
        assert isinstance(step, BenchmarkStep)
        assert step.command
        assert step.purpose
        assert step.collapses_confidence_on
        assert step.priority >= 1
        assert step.label


def test_plan_has_at_least_five_steps(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    assert len(bp.steps) >= 5


def test_plan_has_rationale(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    assert len(bp.rationale) > 20


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


def test_steps_are_ordered_by_priority(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    priorities = [s.priority for s in bp.steps]
    assert priorities == sorted(priorities)


def test_first_step_is_saturation_sweep(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    first = bp.steps[0]
    assert "saturation" in first.label.lower() or "c=16" in first.label.lower()


def test_scale_test_comes_before_soak(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    labels = [s.label.lower() for s in bp.steps]
    scale_idx = next(i for i, l in enumerate(labels) if "scale" in l)
    soak_idx = next(i for i, l in enumerate(labels) if "soak" in l)
    assert scale_idx < soak_idx


# ---------------------------------------------------------------------------
# Command validity — all commands are valid run_bench.py invocations
# ---------------------------------------------------------------------------


def test_all_commands_start_with_run_bench(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    for step in bp.steps:
        assert step.command.startswith("python collect/run_bench.py"), \
            f"Bad command: {step.command}"


def test_all_commands_have_isl_flag(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    for step in bp.steps:
        assert "--isl" in step.command, f"Missing --isl in: {step.command}"


def test_all_commands_have_osl_flag(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    for step in bp.steps:
        assert "--osl" in step.command, f"Missing --osl in: {step.command}"


def test_all_commands_have_model_flag(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    for step in bp.steps:
        assert "--model" in step.command, f"Missing --model in: {step.command}"


def test_all_commands_have_concurrency_flag(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    for step in bp.steps:
        assert "--concurrency" in step.command, f"Missing --concurrency in: {step.command}"


def test_all_commands_have_tag_flag(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    for step in bp.steps:
        assert "--tag" in step.command, f"Missing --tag in: {step.command}"


def test_tags_are_unique(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    tags = []
    for step in bp.steps:
        parts = step.command.split()
        tag_idx = parts.index("--tag") + 1
        tags.append(parts[tag_idx])
    assert len(tags) == len(set(tags)), f"Duplicate tags: {tags}"


# ---------------------------------------------------------------------------
# ISL is the real workload ISL in single-replica tests
# ---------------------------------------------------------------------------


def test_saturation_tests_use_workload_isl(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    isl = _infer_isl(golden_estimate)
    saturation_steps = [s for s in bp.steps if "saturation" in s.label.lower()]
    for step in saturation_steps:
        assert f"--isl {isl}" in step.command, \
            f"Expected ISL={isl} in saturation step: {step.command}"


# ---------------------------------------------------------------------------
# Soak / burst durations
# ---------------------------------------------------------------------------


def test_soak_test_has_long_duration(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    soak = next(s for s in bp.steps if "soak" in s.label.lower())
    parts = soak.command.split()
    dur_idx = parts.index("--duration") + 1
    assert int(parts[dur_idx]) >= 3600, "Soak test should run at least 1 hour"


def test_burst_test_has_medium_duration(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    burst = next(s for s in bp.steps if "burst" in s.label.lower())
    parts = burst.command.split()
    dur_idx = parts.index("--duration") + 1
    duration = int(parts[dur_idx])
    assert 300 <= duration <= 1200, f"Burst duration should be 5–20 min, got {duration}s"


# ---------------------------------------------------------------------------
# Chunked-prefill variant only at high ISL
# ---------------------------------------------------------------------------


def test_chunked_prefill_step_present_at_high_isl(golden_estimate):
    # golden estimate has ISL=9000 → should include chunked-prefill variant
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm",
                        include_chunked_prefill_variant=True)
    has_chunked = any("chunked" in s.label.lower() for s in bp.steps)
    assert has_chunked


def test_chunked_prefill_step_absent_at_low_isl(high_conf_estimate):
    # high_conf_estimate has ISL=2048 < 4096 → no chunked-prefill step
    bp = benchmark_plan(high_conf_estimate, "llama-3.1-8b", "l4",
                        include_chunked_prefill_variant=True)
    has_chunked = any("chunked" in s.label.lower() for s in bp.steps)
    assert not has_chunked


def test_chunked_prefill_step_omittable(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm",
                        include_chunked_prefill_variant=False)
    has_chunked = any("chunked" in s.label.lower() for s in bp.steps)
    assert not has_chunked


# ---------------------------------------------------------------------------
# Confidence metadata propagated
# ---------------------------------------------------------------------------


def test_plan_carries_confidence_label(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    assert bp.confidence == "low"


def test_plan_carries_binding_constraint(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    assert bp.binding_constraint == "prefill-bound"


def test_high_confidence_rationale_mentions_scale(high_conf_estimate):
    bp = benchmark_plan(high_conf_estimate, "llama-3.1-8b", "l4")
    assert "scale" in bp.rationale.lower()


def test_low_confidence_rationale_mentions_mfu(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    assert "mfu" in bp.rationale.lower() or "confidence" in bp.rationale.lower()


# ---------------------------------------------------------------------------
# collapses_confidence_on is informative (not empty)
# ---------------------------------------------------------------------------


def test_first_step_collapses_confidence_on_mfu(golden_estimate):
    bp = benchmark_plan(golden_estimate, "gpt-oss-20b", "h100_sxm")
    first = bp.steps[0]
    combined = first.collapses_confidence_on.lower()
    assert "mfu" in combined or "prefill" in combined or "confidence" in combined


# ---------------------------------------------------------------------------
# _infer_isl / _infer_osl helpers
# ---------------------------------------------------------------------------


def test_infer_isl_roundtrip(golden_estimate):
    isl = _infer_isl(golden_estimate)
    assert abs(isl - 9000) <= 1  # should recover 9000 exactly


def test_infer_osl_roundtrip(golden_estimate):
    osl = _infer_osl(golden_estimate)
    assert abs(osl - 500) <= 1
