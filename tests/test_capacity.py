"""Phase 1 tests for planner/capacity.py.

Includes:
  - GOLDEN TEST: gpt-oss-20b @ H100, 100M req/day, ISL 9000, OSL 500 (prefill-bound, low confidence)
  - AGNOSTIC TEST: full inline spec + param-only rough spec, neither in seed catalog
  - Unit tests for each pipeline function
"""
import math
import pytest

from planner.catalog import CatalogError, get_gpu, get_model, resolve_model, resolve_gpu
from planner.capacity import (
    CapacityEstimate,
    KvBudget,
    Traffic,
    TtftEstimate,
    decode_ceiling,
    kv_budget,
    normalize_traffic,
    plan,
    prefill_ceiling,
    size_replicas,
    ttft_estimate,
    confidence,
)
import planner.catalog as _catalog_module


@pytest.fixture(autouse=True)
def reset_catalog():
    _catalog_module._catalog = None
    yield
    _catalog_module._catalog = None


# ============================================================================
# GOLDEN TEST
# Input: 100M req/day, peak 3x, ISL 9000, OSL 500
#        gpt-oss-20b, h100_sxm, mxfp4, tp=1, realtime
# ============================================================================


@pytest.fixture
def golden_estimate():
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


def test_golden_avg_rps_within_1pct(golden_estimate):
    # 100M / 86400 ≈ 1157.4
    assert abs(golden_estimate.traffic.avg_rps - 1157.4) / 1157.4 < 0.01


def test_golden_peak_rps_within_1pct(golden_estimate):
    # 1157.4 * 3 ≈ 3472.2
    assert abs(golden_estimate.traffic.peak_rps - 3472.2) / 3472.2 < 0.01


def test_golden_input_tps_avg_within_1pct(golden_estimate):
    # 1157.4 * 9000 ≈ 10,416,667
    expected = 1157.4 * 9000
    assert abs(golden_estimate.traffic.input_tps_avg - expected) / expected < 0.01


def test_golden_output_tps_avg_within_1pct(golden_estimate):
    # 1157.4 * 500 ≈ 578,704
    expected = 1157.4 * 500
    assert abs(golden_estimate.traffic.output_tps_avg - expected) / expected < 0.01


def test_golden_prefill_bound(golden_estimate):
    assert golden_estimate.binding_constraint == "prefill-bound"


def test_golden_confidence_low(golden_estimate):
    # No H100 / gpt-oss-20b anchor seeded → LOW
    assert golden_estimate.confidence == "low"


def test_golden_replica_range_low_lt_high(golden_estimate):
    assert golden_estimate.replicas_low < golden_estimate.replicas_high


def test_golden_replica_range_wide(golden_estimate):
    # LOW band = ±50%: replicas_high should be ≥ 1.4× replicas_low
    ratio = golden_estimate.replicas_high / golden_estimate.replicas_low
    assert ratio >= 1.4, f"Range not wide enough for LOW confidence: ratio={ratio:.2f}"


def test_golden_validate_warning_present(golden_estimate):
    combined = " ".join(golden_estimate.warnings).lower()
    assert "validate" in combined and ("live gpu" in combined or "live" in combined)


def test_golden_chunked_prefill_warning_present(golden_estimate):
    combined = " ".join(golden_estimate.warnings).lower()
    assert "chunked prefill" in combined


def test_golden_replicas_positive(golden_estimate):
    assert golden_estimate.replicas >= 1
    assert golden_estimate.replicas_low >= 1
    assert golden_estimate.replicas_high > golden_estimate.replicas_low


# ============================================================================
# AGNOSTIC TEST — model AND GPU not in seed catalog
# ============================================================================


CUSTOM_GPU_SPEC = {
    "name": "custom-h100-variant",
    "display_name": "Custom H100 Variant",
    "mem_gb": 80,
    "hbm_bandwidth_gbps": 3350,
    "peak_flops": {"fp16": 989, "bf16": 989, "fp8": 1979, "mxfp4": 1979},
    "default_mfu_prefill": 0.40,
    "default_bw_efficiency_decode": 0.70,
}

CUSTOM_MODEL_FULL_SPEC = {
    "name": "custom-dense-13b",
    "display_name": "Custom Dense 13B",
    "is_moe": False,
    "total_params": 13_000_000_000,
    "active_params": 13_000_000_000,
    "num_layers": 40,
    "d_model": 5120,
    "num_q_heads": 40,
    "num_kv_heads": 8,
    "head_dim": 128,
    "native_dtype": "fp8",
    "weight_bytes_per_param": 1.0,
    "kv_dtype_bytes": 1,
}


def test_agnostic_full_inline_spec_produces_valid_estimate():
    model = resolve_model(CUSTOM_MODEL_FULL_SPEC)
    gpu = resolve_gpu(CUSTOM_GPU_SPEC)
    est = plan(
        requests_per_day=1_000_000,
        peak_multiplier=2.0,
        isl=2048,
        osl=256,
        ttft_slo_ms=3000.0,
        model=model,
        gpu=gpu,
        dtype="fp8",
        tp=1,
        traffic_class="mixed",
    )
    assert isinstance(est, CapacityEstimate)
    assert est.replicas >= 1
    assert est.replicas_low < est.replicas_high
    assert est.binding_constraint in ("prefill-bound", "decode-bound", "kv-memory-bound")
    assert est.confidence in ("high", "medium", "low")


def test_agnostic_full_inline_spec_geometry_known():
    model = resolve_model(CUSTOM_MODEL_FULL_SPEC)
    assert model.geometry_source == "known"


def test_agnostic_rough_spec_produces_valid_estimate():
    rough_spec = {
        "name": "rough-20b-custom",
        "total_params": 20_000_000_000,
        "native_dtype": "bf16",
    }
    model = resolve_model(rough_spec)
    gpu = resolve_gpu(CUSTOM_GPU_SPEC)
    est = plan(
        requests_per_day=500_000,
        peak_multiplier=2.0,
        isl=1024,
        osl=128,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="bf16",
        tp=1,
        traffic_class="batch",
    )
    assert isinstance(est, CapacityEstimate)
    assert est.replicas >= 1


def test_agnostic_rough_spec_geometry_source_estimated():
    rough_spec = {
        "name": "rough-20b-check",
        "total_params": 20_000_000_000,
        "native_dtype": "bf16",
    }
    model = resolve_model(rough_spec)
    assert model.geometry_source == "estimated"


def test_agnostic_rough_spec_confidence_downgraded():
    """Rough-spec model can be at most MEDIUM confidence, never HIGH."""
    rough_spec = {
        "name": "rough-8b-l4-test",
        "total_params": 8_030_000_000,
        "native_dtype": "bf16",
    }
    model = resolve_model(rough_spec)
    gpu = get_gpu("l4")
    # l4 has anchors for llama-3.1-8b/fp8 — rough spec of similar size should NOT inherit HIGH
    est = plan(
        requests_per_day=100_000,
        peak_multiplier=2.0,
        isl=512,
        osl=128,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="fp8",
        tp=1,
        traffic_class="batch",
    )
    assert est.confidence in ("medium", "low"), (
        f"Rough spec should never be HIGH confidence, got: {est.confidence}"
    )


def test_agnostic_rough_spec_has_geometry_warning():
    rough_spec = {
        "name": "rough-20b-warn",
        "total_params": 20_000_000_000,
        "native_dtype": "bf16",
    }
    model = resolve_model(rough_spec)
    gpu = resolve_gpu(CUSTOM_GPU_SPEC)
    est = plan(
        requests_per_day=100_000,
        peak_multiplier=2.0,
        isl=512,
        osl=128,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="bf16",
        tp=1,
        traffic_class="batch",
    )
    combined = " ".join(est.warnings).lower()
    assert "geometry" in combined and "estimated" in combined


# ============================================================================
# Unit tests — normalize_traffic
# ============================================================================


def test_normalize_traffic_avg_rps():
    t = normalize_traffic(86_400, 1.0, 1024, 128)
    assert t.avg_rps == pytest.approx(1.0, rel=1e-6)
    assert t.peak_rps == pytest.approx(1.0, rel=1e-6)


def test_normalize_traffic_tps_values():
    t = normalize_traffic(86_400, 3.0, 1000, 100)
    assert t.input_tps_avg == pytest.approx(1000.0, rel=1e-6)
    assert t.output_tps_avg == pytest.approx(100.0, rel=1e-6)
    assert t.input_tps_peak == pytest.approx(3000.0, rel=1e-6)
    assert t.output_tps_peak == pytest.approx(300.0, rel=1e-6)


def test_normalize_traffic_total_tokens():
    t = normalize_traffic(1_000_000, 1.0, 500, 100)
    assert t.total_tokens_day == pytest.approx(600_000_000, rel=1e-6)


# ============================================================================
# Unit tests — kv_budget
# ============================================================================


def test_kv_budget_llama_8b_l4():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    kb = kv_budget(gpu, model, "fp8", 2048, 128)
    # l4 has 24 GB; llama-8b weights ≈ 8.03B * 2 bytes = 16.06 GB
    # usable = 24e9 * 0.9 = 21.6 GB
    # kv_budget = 21.6 - 16.06 - 2 = 3.54 GB
    assert kb.kv_cache_budget_bytes > 0
    assert kb.max_concurrent_seqs >= 1


def test_kv_budget_model_fits_h100():
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    kb = kv_budget(gpu, model, "mxfp4", 9000, 500)
    assert kb.max_concurrent_seqs >= 1


def test_kv_budget_model_does_not_fit_raises():
    # llama-3.1-70b at bf16 is ~140 GB — won't fit on L4 (24 GB)
    model = get_model("llama-3.1-70b")
    gpu = get_gpu("l4")
    with pytest.raises(CatalogError, match="Use a larger tp"):
        kv_budget(gpu, model, "bf16", 2048, 128)


def test_kv_bytes_per_token_formula():
    model = get_model("llama-3.1-8b")
    # 2 * 32 layers * 8 kv_heads * 128 head_dim * 1 kv_dtype_bytes = 65536
    expected = 2 * 32 * 8 * 128 * 1
    assert model.kv_bytes_per_token == expected


# ============================================================================
# Unit tests — prefill_ceiling
# ============================================================================


def test_prefill_ceiling_increases_with_mfu():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    low = prefill_ceiling(gpu, model, "fp8", 2048, 0.30)
    high = prefill_ceiling(gpu, model, "fp8", 2048, 0.60)
    assert high > low


def test_prefill_ceiling_decreases_with_longer_isl():
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    short = prefill_ceiling(gpu, model, "mxfp4", 1000, 0.40)
    long_ = prefill_ceiling(gpu, model, "mxfp4", 9000, 0.40)
    assert short > long_, "Longer ISL must reduce prefill ceiling due to attention FLOPs"


def test_prefill_ceiling_positive():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    tps = prefill_ceiling(gpu, model, "fp8", 512, 0.40)
    assert tps > 0


# ============================================================================
# Unit tests — decode_ceiling
# ============================================================================


def test_decode_ceiling_increases_with_batch():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    small = decode_ceiling(gpu, model, "fp8", 8, 1024, 0.70)
    large = decode_ceiling(gpu, model, "fp8", 64, 1024, 0.70)
    assert large > small


def test_decode_ceiling_decreases_with_avg_ctx():
    """Larger avg_ctx means more KV reads → lower decode throughput."""
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    short_ctx = decode_ceiling(gpu, model, "mxfp4", 32, 1000, 0.70)
    long_ctx = decode_ceiling(gpu, model, "mxfp4", 32, 9000, 0.70)
    assert short_ctx > long_ctx


def test_decode_ceiling_positive():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    tps = decode_ceiling(gpu, model, "fp8", 16, 512, 0.65)
    assert tps > 0


# ============================================================================
# Unit tests — confidence
# ============================================================================


def test_confidence_high_for_seeded_l4_anchor():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    conf = confidence(model, gpu, "fp8", 2048, 128, 10)
    assert conf.level == "high"
    assert conf.anchor_matched is True


def test_confidence_low_for_unknown_model_gpu():
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    conf = confidence(model, gpu, "mxfp4", 9000, 500, 100)
    assert conf.level == "low"
    assert conf.anchor_matched is False


def test_confidence_estimated_geometry_downgrade():
    rough_spec = {
        "name": "rough-8b",
        "total_params": 8_000_000_000,
        "native_dtype": "fp8",
    }
    model = resolve_model(rough_spec)
    gpu = get_gpu("l4")
    # Even though l4/fp8 anchors exist for llama-8b, rough spec should downgrade
    conf = confidence(model, gpu, "fp8", 2048, 128, 10)
    # rough spec → the name "rough-8b" won't match the anchor's model "llama-3.1-8b"
    # so it would be LOW anyway; but even if there WERE anchors, it must not be HIGH
    assert conf.level in ("medium", "low")


def test_confidence_mfu_from_anchor_when_available():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    conf = confidence(model, gpu, "fp8", 2048, 128, 10)
    # Anchor derived_mfu_prefill is null in seeded data → falls back to GPU default
    assert conf.mfu_used == gpu.default_mfu_prefill


# ============================================================================
# Unit tests — ttft_estimate
# ============================================================================


def test_ttft_compute_only_no_queue():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    tps = prefill_ceiling(gpu, model, "fp8", 2048, 0.40) * 1  # tp=1
    result = ttft_estimate(gpu, model, "fp8", 2048, tps, 0.0, 5000.0)
    assert result.ttft_queue_ms == pytest.approx(0.0)
    assert result.ttft_ms == pytest.approx(result.ttft_compute_ms)


def test_ttft_queue_grows_with_utilization():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    tps = 50_000.0
    low_util = ttft_estimate(gpu, model, "fp8", 2048, tps, 0.20, 5000.0)
    high_util = ttft_estimate(gpu, model, "fp8", 2048, tps, 0.80, 5000.0)
    assert high_util.ttft_ms > low_util.ttft_ms


def test_ttft_slo_met_flag():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    tps = 1_000_000.0  # very fast
    result = ttft_estimate(gpu, model, "fp8", 512, tps, 0.10, 5000.0)
    assert result.slo_met is True
    assert result.slo_breach_reason is None


def test_ttft_slo_breach_flag():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    tps = 100.0  # extremely slow prefill to force SLO breach
    result = ttft_estimate(gpu, model, "fp8", 2048, tps, 0.10, 500.0)
    assert result.slo_met is False
    assert result.slo_breach_reason is not None


# ============================================================================
# Unit tests — size_replicas
# ============================================================================


def test_size_replicas_prefill_bound():
    traffic = normalize_traffic(100_000_000, 3.0, 9000, 500)
    sz = size_replicas(
        traffic=traffic,
        prefill_tps_gpu=90_000.0,      # low prefill → many replicas needed
        decode_tps_gpu=500_000.0,      # high decode → few replicas needed
        max_concurrent_seqs=200,
        tp=1,
        traffic_class="realtime",
        isl=9000,
        osl=500,
    )
    assert sz["binding_constraint"] == "prefill-bound"


def test_size_replicas_decode_bound():
    traffic = normalize_traffic(1_000_000, 2.0, 256, 2000)  # high output
    sz = size_replicas(
        traffic=traffic,
        prefill_tps_gpu=500_000.0,     # fast prefill
        decode_tps_gpu=500.0,          # very slow decode
        max_concurrent_seqs=512,
        tp=1,
        traffic_class="realtime",
        isl=256,
        osl=2000,
    )
    assert sz["binding_constraint"] == "decode-bound"


def test_size_replicas_headroom_applied():
    # 80M req/day, isl=512 → input_tps_peak ≈ 474k → base ≈ 5 replicas at 100k tps/GPU
    # ceil(5 * 1.40) = 7  vs  ceil(5 * 1.10) = 6  → realtime > batch
    traffic = normalize_traffic(80_000_000, 1.0, 512, 128)
    sz_realtime = size_replicas(traffic, 100_000.0, 100_000.0, 64, 1, "realtime", 512, 128)
    sz_batch = size_replicas(traffic, 100_000.0, 100_000.0, 64, 1, "batch", 512, 128)
    assert sz_realtime["replicas"] > sz_batch["replicas"]


def test_size_replicas_tp_scales_per_replica_capacity():
    traffic = normalize_traffic(1_000_000, 1.0, 512, 128)
    sz_tp1 = size_replicas(traffic, 50_000.0, 50_000.0, 64, tp=1, traffic_class="batch", isl=512, osl=128)
    sz_tp4 = size_replicas(traffic, 50_000.0, 50_000.0, 64, tp=4, traffic_class="batch", isl=512, osl=128)
    # tp=4 means each replica is 4x more capable → fewer replicas
    assert sz_tp4["replicas"] <= sz_tp1["replicas"]


# ============================================================================
# Integration: CLI-equivalent paths
# ============================================================================


def test_plan_produces_assumptions_list(golden_estimate):
    assert len(golden_estimate.assumptions) > 0
    # MFU and bandwidth efficiency should be documented
    combined = " ".join(golden_estimate.assumptions).lower()
    assert "mfu" in combined
    assert "bandwidth" in combined


def test_plan_llama_l4_has_high_confidence():
    """Seeded anchors should yield high confidence for llama-3.1-8b on l4."""
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    est = plan(
        requests_per_day=100_000,
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
    assert est.confidence == "high"
    assert est.anchor_matched is True
