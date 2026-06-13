"""Phase 2 tests for planner/cost.py."""
import pytest

from planner.catalog import CatalogError, get_gpu, get_model
from planner.capacity import plan
from planner.cost import CostEstimate, CostVariant, compute_cost
import planner.catalog as _catalog_module


@pytest.fixture(autouse=True)
def reset_catalog():
    _catalog_module._catalog = None
    yield
    _catalog_module._catalog = None


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


@pytest.fixture
def small_estimate():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    return plan(
        requests_per_day=1_000_000,
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
# Structural tests
# ---------------------------------------------------------------------------


def test_compute_cost_returns_cost_estimate(golden_estimate):
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    assert isinstance(cost, CostEstimate)


def test_cost_has_both_variants(golden_estimate):
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    assert isinstance(cost.on_demand, CostVariant)
    assert isinstance(cost.reserved, CostVariant)


# ---------------------------------------------------------------------------
# Formula correctness
# ---------------------------------------------------------------------------


def test_gpu_hours_day_formula(golden_estimate):
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    expected_gpu_hours = golden_estimate.replicas * 1 * 24
    assert cost.on_demand.gpu_hours_day == pytest.approx(expected_gpu_hours)


def test_gpu_hours_day_scales_with_tp(golden_estimate):
    cost_tp1 = compute_cost(golden_estimate, "h100_sxm", tp=1)
    cost_tp4 = compute_cost(golden_estimate, "h100_sxm", tp=4)
    assert cost_tp4.on_demand.gpu_hours_day == pytest.approx(cost_tp1.on_demand.gpu_hours_day * 4)


def test_cost_day_formula(golden_estimate):
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    od = cost.on_demand
    assert od.cost_day_usd == pytest.approx(od.gpu_hours_day * od.price_usd_per_gpu_hour)


def test_cost_month_is_30x_day(golden_estimate):
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    assert cost.on_demand.cost_month_usd == pytest.approx(cost.on_demand.cost_day_usd * 30)
    assert cost.reserved.cost_month_usd == pytest.approx(cost.reserved.cost_day_usd * 30)


def test_cost_per_1m_tokens_formula(golden_estimate):
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    total_tok = golden_estimate.traffic.total_tokens_day
    expected = cost.on_demand.cost_day_usd / (total_tok / 1e6)
    assert cost.on_demand.cost_per_1m_tokens_usd == pytest.approx(expected)


def test_cost_per_request_formula(golden_estimate):
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    rpd = golden_estimate.traffic.requests_per_day
    expected = cost.on_demand.cost_day_usd / rpd
    assert cost.on_demand.cost_per_request_usd == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Reserved is cheaper than on-demand
# ---------------------------------------------------------------------------


def test_reserved_cheaper_than_on_demand(golden_estimate):
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    assert cost.reserved.cost_day_usd < cost.on_demand.cost_day_usd
    assert cost.reserved.price_usd_per_gpu_hour < cost.on_demand.price_usd_per_gpu_hour


# ---------------------------------------------------------------------------
# Prefill / decode split
# ---------------------------------------------------------------------------


def test_breakdown_fractions_sum_to_one(golden_estimate):
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    total = cost.breakdown.prefill_fraction + cost.breakdown.decode_fraction
    assert total == pytest.approx(1.0)


def test_breakdown_cost_sums_to_total(golden_estimate):
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    total = cost.breakdown.prefill_cost_day_usd + cost.breakdown.decode_cost_day_usd
    assert total == pytest.approx(cost.on_demand.cost_day_usd)


def test_prefill_bound_estimate_has_high_prefill_fraction(golden_estimate):
    # golden test is prefill-bound → prefill replicas >> decode replicas
    assert golden_estimate.binding_constraint == "prefill-bound"
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    assert cost.breakdown.prefill_fraction > cost.breakdown.decode_fraction


# ---------------------------------------------------------------------------
# Price overrides
# ---------------------------------------------------------------------------


def test_override_on_demand_price(golden_estimate):
    custom_rate = 5.00
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1,
                        override_on_demand_usd_per_hour=custom_rate)
    assert cost.on_demand.price_usd_per_gpu_hour == custom_rate


def test_override_reserved_price(golden_estimate):
    custom_rate = 3.50
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1,
                        override_reserved_usd_per_hour=custom_rate)
    assert cost.reserved.price_usd_per_gpu_hour == custom_rate


def test_unknown_gpu_without_override_raises(golden_estimate):
    with pytest.raises(CatalogError, match="No cost profile"):
        compute_cost(golden_estimate, "nonexistent-gpu-xyz", tp=1)


def test_unknown_gpu_with_override_succeeds(golden_estimate):
    cost = compute_cost(
        golden_estimate, "custom-a100-variant", tp=1,
        override_on_demand_usd_per_hour=2.50,
        override_reserved_usd_per_hour=1.75,
    )
    assert cost.on_demand.cost_day_usd > 0
    assert "caller override" in cost.price_source


# ---------------------------------------------------------------------------
# All values are positive
# ---------------------------------------------------------------------------


def test_all_cost_values_positive(small_estimate):
    cost = compute_cost(small_estimate, "l4", tp=1)
    od = cost.on_demand
    assert od.cost_day_usd > 0
    assert od.cost_month_usd > 0
    assert od.cost_per_1m_tokens_usd > 0
    assert od.cost_per_request_usd > 0


# ---------------------------------------------------------------------------
# Warnings always present
# ---------------------------------------------------------------------------


def test_cost_always_has_disclaimer_warning(golden_estimate):
    cost = compute_cost(golden_estimate, "h100_sxm", tp=1)
    combined = " ".join(cost.warnings).lower()
    assert "verify" in combined or "contract" in combined
