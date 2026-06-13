"""Phase 0 acceptance tests for planner/catalog.py."""

import shutil
import pytest

from planner.catalog import (
    CatalogError,
    load_catalog,
    get_gpu,
    get_model,
    find_anchors,
    resolve_model,
    resolve_gpu,
    register_model,
    register_gpu,
)
import planner.catalog as _catalog_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_catalog_singleton():
    """Force catalog reload between tests so register_* changes are visible."""
    _catalog_module._catalog = None
    yield
    _catalog_module._catalog = None


@pytest.fixture
def clean_user_catalog(tmp_path, monkeypatch):
    """Redirect the user-catalog dir to a temp path so tests don't pollute ~."""
    monkeypatch.setattr(_catalog_module, "_USER_CATALOG", tmp_path / "user-catalog")
    _catalog_module._catalog = None
    yield tmp_path / "user-catalog"
    _catalog_module._catalog = None


# ---------------------------------------------------------------------------
# load_catalog — parses all seeded YAMLs without validation errors
# ---------------------------------------------------------------------------


def test_load_catalog_parses_all_gpus():
    cat = load_catalog()
    assert len(cat.gpus) >= 5  # h100, h200, a100, l40s, l4
    assert "h100_sxm" in cat.gpus
    assert "l4" in cat.gpus


def test_load_catalog_parses_all_models():
    cat = load_catalog()
    assert len(cat.models) >= 3
    assert "gpt-oss-20b" in cat.models
    assert "llama-3.1-8b" in cat.models
    assert "llama-3.1-70b" in cat.models


def test_load_catalog_parses_costs():
    cat = load_catalog()
    assert "h100_sxm" in cat.costs
    assert cat.costs["h100_sxm"].on_demand_usd_per_hour > 0


def test_load_catalog_parses_runtimes():
    cat = load_catalog()
    assert "vllm" in cat.runtimes
    assert "sglang" in cat.runtimes


def test_load_catalog_parses_anchors():
    cat = load_catalog()
    assert len(cat.anchors) >= 6
    models = {a.model for a in cat.anchors}
    assert "llama-3.1-8b" in models


def test_gpu_profile_fields():
    gpu = get_gpu("h100_sxm")
    assert gpu.mem_gb == 80
    assert gpu.hbm_bandwidth_gbps == 3350
    assert gpu.peak_flops.fp8 == 1979
    assert gpu.default_mfu_prefill == 0.40


def test_model_profile_moe_fields():
    model = get_model("gpt-oss-20b")
    assert model.is_moe is True
    assert model.total_params > model.active_params
    assert model.geometry_source == "known"


def test_model_profile_dense_fields():
    model = get_model("llama-3.1-8b")
    assert model.is_moe is False
    assert model.total_params == model.active_params


# ---------------------------------------------------------------------------
# CatalogError on unknown names — never a bare KeyError
# ---------------------------------------------------------------------------


def test_get_gpu_unknown_raises_catalog_error():
    with pytest.raises(CatalogError, match="Unknown GPU"):
        get_gpu("nonexistent-gpu-xyz")


def test_get_model_unknown_raises_catalog_error():
    with pytest.raises(CatalogError, match="Unknown model"):
        get_model("nonexistent-model-xyz")


def test_catalog_error_not_key_error():
    with pytest.raises(CatalogError):
        get_gpu("totally-fake")


# ---------------------------------------------------------------------------
# resolve_model — three paths
# ---------------------------------------------------------------------------


def test_resolve_model_by_name():
    m = resolve_model("llama-3.1-8b")
    assert m.name == "llama-3.1-8b"
    assert m.geometry_source == "known"


def test_resolve_model_full_inline_spec():
    spec = {
        "name": "test-dense-7b",
        "display_name": "Test Dense 7B",
        "is_moe": False,
        "total_params": 7_000_000_000,
        "active_params": 7_000_000_000,
        "num_layers": 32,
        "d_model": 4096,
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "native_dtype": "bf16",
        "weight_bytes_per_param": 2.0,
        "kv_dtype_bytes": 2,
    }
    m = resolve_model(spec)
    assert m.name == "test-dense-7b"
    assert m.geometry_source == "known"
    assert m.num_layers == 32


def test_resolve_model_rough_spec_geometry_estimated():
    spec = {
        "name": "rough-20b",
        "total_params": 20_000_000_000,
        "native_dtype": "bf16",
    }
    m = resolve_model(spec)
    assert m.geometry_source == "estimated"
    assert m.num_layers > 0
    assert m.d_model > 0
    assert m.num_q_heads > 0


def test_resolve_model_rough_spec_moe_active_params():
    spec = {
        "name": "rough-moe-50b",
        "total_params": 50_000_000_000,
        "active_params": 5_000_000_000,
        "native_dtype": "mxfp4",
    }
    m = resolve_model(spec)
    assert m.geometry_source == "estimated"
    assert m.active_params == 5_000_000_000
    assert m.total_params == 50_000_000_000


def test_resolve_model_rough_spec_missing_total_params_raises():
    with pytest.raises(CatalogError, match="total_params"):
        resolve_model({"name": "bad-spec", "native_dtype": "bf16"})


# ---------------------------------------------------------------------------
# resolve_gpu — two paths (name and full spec; no rough path)
# ---------------------------------------------------------------------------


def test_resolve_gpu_by_name():
    g = resolve_gpu("h200_sxm")
    assert g.mem_gb == 141


def test_resolve_gpu_full_inline_spec():
    spec = {
        "name": "custom-h100",
        "display_name": "Custom H100",
        "mem_gb": 80,
        "hbm_bandwidth_gbps": 3350,
        "peak_flops": {"fp16": 989, "bf16": 989, "fp8": 1979},
    }
    g = resolve_gpu(spec)
    assert g.name == "custom-h100"
    assert g.mem_gb == 80


def test_resolve_gpu_missing_required_field_raises():
    spec = {
        "name": "incomplete-gpu",
        "mem_gb": 80,
        # missing hbm_bandwidth_gbps and peak_flops
    }
    with pytest.raises(CatalogError, match="missing required fields"):
        resolve_gpu(spec)


# ---------------------------------------------------------------------------
# find_anchors — L4/FP8 rows + no-match returns []
# ---------------------------------------------------------------------------


def test_find_anchors_returns_l4_fp8_matches():
    anchors = find_anchors(
        model="llama-3.1-8b",
        gpu="l4",
        dtype="fp8",
        isl=2048,
        osl=128,
        concurrency=10,
    )
    assert len(anchors) >= 1
    assert all(a.model == "llama-3.1-8b" for a in anchors)
    assert all(a.gpu == "l4" for a in anchors)


def test_find_anchors_no_match_returns_empty():
    anchors = find_anchors(
        model="gpt-oss-20b",
        gpu="h100_sxm",
        dtype="mxfp4",
        isl=9000,
        osl=500,
        concurrency=100,
    )
    assert anchors == []


def test_find_anchors_isl_tolerance():
    # ISL 2200 is within 20% of 2048 → should match
    anchors = find_anchors(
        model="llama-3.1-8b",
        gpu="l4",
        dtype="fp8",
        isl=2200,
        osl=128,
        concurrency=10,
    )
    assert len(anchors) >= 1


def test_find_anchors_isl_too_far_returns_empty():
    # ISL 10000 is far from 512/2048/4096 → no match
    anchors = find_anchors(
        model="llama-3.1-8b",
        gpu="l4",
        dtype="fp8",
        isl=10000,
        osl=128,
        concurrency=10,
    )
    assert anchors == []


# ---------------------------------------------------------------------------
# register_model / register_gpu — round-trip through user catalog
# ---------------------------------------------------------------------------


def test_register_model_round_trip(clean_user_catalog):
    spec = {
        "name": "my-custom-13b",
        "display_name": "My Custom 13B",
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
    registered = register_model(spec)
    assert registered.name == "my-custom-13b"

    # After registration the singleton is cleared; get_model should find it.
    retrieved = get_model("my-custom-13b")
    assert retrieved.num_layers == 40
    assert retrieved.geometry_source == "known"


def test_register_gpu_round_trip(clean_user_catalog):
    spec = {
        "name": "mi300x",
        "display_name": "AMD MI300X",
        "mem_gb": 192,
        "hbm_bandwidth_gbps": 5300,
        "peak_flops": {"fp16": 1300, "bf16": 1300, "fp8": 2600},
    }
    registered = register_gpu(spec)
    assert registered.name == "mi300x"

    retrieved = get_gpu("mi300x")
    assert retrieved.mem_gb == 192
    assert retrieved.hbm_bandwidth_gbps == 5300


def test_register_model_estimated_geometry_persists(clean_user_catalog):
    spec = {
        "name": "rough-7b-registered",
        "total_params": 7_000_000_000,
        "native_dtype": "bf16",
    }
    registered = register_model(spec)
    assert registered.geometry_source == "estimated"

    retrieved = get_model("rough-7b-registered")
    assert retrieved.geometry_source == "estimated"


# ---------------------------------------------------------------------------
# kv_bytes_per_token property
# ---------------------------------------------------------------------------


def test_kv_bytes_per_token_llama_8b():
    m = get_model("llama-3.1-8b")
    # 2 * 32 layers * 8 kv_heads * 128 head_dim * 1 kv_dtype_bytes = 65536
    expected = 2 * m.num_layers * m.num_kv_heads * m.head_dim * m.kv_dtype_bytes
    assert m.kv_bytes_per_token == expected


def test_resident_weights_bytes_uses_override():
    m = get_model("gpt-oss-20b")
    # resident_weights_gb is set to 13.0 for gpt-oss-20b
    assert m.resident_weights_bytes == 13.0 * 1e9


def test_resident_weights_bytes_fallback():
    m = get_model("llama-3.1-8b")
    # no resident_weights_gb → uses total_params * weight_bytes_per_param
    assert m.resident_weights_bytes == m.total_params * m.weight_bytes_per_param
