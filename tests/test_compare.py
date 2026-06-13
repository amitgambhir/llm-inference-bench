"""Phase 5 tests for planner/compare.py."""
import pytest

from planner.catalog import get_gpu, get_model
from planner.capacity import plan
from planner.cost import compute_cost
from planner.compare import (
    ComparisonEntry,
    ComparisonResult,
    Winner,
    compare,
)
import planner.catalog as _catalog_module


@pytest.fixture(autouse=True)
def reset_catalog():
    _catalog_module._catalog = None
    yield
    _catalog_module._catalog = None


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def l4_entry():
    """HIGH confidence: llama-3.1-8b / L4 / fp8 — backed by seeded anchors."""
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    est = plan(
        requests_per_day=500_000,
        peak_multiplier=2.0,
        isl=2048, osl=128,
        ttft_slo_ms=5000.0,
        model=model, gpu=gpu, dtype="fp8",
        tp=1, traffic_class="batch",
    )
    cost = compute_cost(est, "l4", tp=1)
    return ComparisonEntry(
        label="l4-fp8",
        estimate=est,
        cost=cost,
        gpu_name="l4",
        model_name="llama-3.1-8b",
        dtype="fp8",
    )


@pytest.fixture
def h100_entry():
    """LOW confidence: gpt-oss-20b / H100 / mxfp4 — no anchor seeded."""
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    est = plan(
        requests_per_day=100_000_000,
        peak_multiplier=3.0,
        isl=9000, osl=500,
        ttft_slo_ms=2000.0,
        model=model, gpu=gpu, dtype="mxfp4",
        tp=1, traffic_class="realtime",
    )
    cost = compute_cost(est, "h100_sxm", tp=1)
    return ComparisonEntry(
        label="h100-mxfp4",
        estimate=est,
        cost=cost,
        gpu_name="h100_sxm",
        model_name="gpt-oss-20b",
        dtype="mxfp4",
    )


@pytest.fixture
def two_entries(l4_entry, h100_entry):
    return [l4_entry, h100_entry]


# ── Return type and structure ─────────────────────────────────────────────────


def test_compare_returns_result_object(two_entries):
    result = compare(two_entries)
    assert isinstance(result, ComparisonResult)


def test_compare_preserves_entries(two_entries):
    result = compare(two_entries)
    assert len(result.entries) == 2


def test_compare_returns_three_winners(two_entries):
    result = compare(two_entries)
    assert isinstance(result.cheapest, Winner)
    assert isinstance(result.safest, Winner)
    assert isinstance(result.best_latency, Winner)


def test_compare_winner_dimensions(two_entries):
    result = compare(two_entries)
    assert result.cheapest.dimension == "cheapest"
    assert result.safest.dimension == "safest"
    assert result.best_latency.dimension == "best_latency"


def test_compare_winners_have_tradeoffs(two_entries):
    result = compare(two_entries)
    for w in (result.cheapest, result.safest, result.best_latency):
        assert len(w.tradeoff) > 10
        assert len(w.metric_value) > 2


def test_compare_winner_labels_are_entry_labels(two_entries):
    result = compare(two_entries)
    valid_labels = {e.label for e in two_entries}
    assert result.cheapest.entry_label in valid_labels
    assert result.safest.entry_label in valid_labels
    assert result.best_latency.entry_label in valid_labels


# ── Safest: highest confidence ────────────────────────────────────────────────


def test_safest_is_high_confidence_over_low(two_entries, l4_entry, h100_entry):
    result = compare(two_entries)
    # l4_entry is HIGH confidence; h100_entry is LOW — safest must be l4
    assert result.safest.entry_label == l4_entry.label


def test_safest_tradeoff_mentions_confidence(two_entries):
    result = compare(two_entries)
    assert "confidence" in result.safest.tradeoff.lower()


def test_safest_metric_includes_confidence_label(two_entries):
    result = compare(two_entries)
    conf_words = {"HIGH", "MEDIUM", "LOW"}
    assert any(w in result.safest.metric_value for w in conf_words)


# ── Cheapest: lowest cost/replicas ────────────────────────────────────────────


def test_cheapest_has_dollar_sign_when_cost_available(two_entries):
    result = compare(two_entries)
    assert "$" in result.cheapest.metric_value


def test_cheapest_tradeoff_mentions_other_entry(two_entries, l4_entry, h100_entry):
    result = compare(two_entries)
    other_label = h100_entry.label if result.cheapest.entry_label == l4_entry.label else l4_entry.label
    assert other_label in result.cheapest.tradeoff


# ── Best latency ──────────────────────────────────────────────────────────────


def test_best_latency_metric_includes_ms(two_entries):
    result = compare(two_entries)
    assert "ms" in result.best_latency.metric_value


def test_best_latency_tradeoff_mentions_binding_constraint(two_entries):
    result = compare(two_entries)
    assert "bound" in result.best_latency.tradeoff.lower()


def test_best_latency_tradeoff_has_caveat(two_entries):
    result = compare(two_entries)
    # M/M/1 caveat must appear
    assert "heuristic" in result.best_latency.tradeoff.lower() or "validate" in result.best_latency.tradeoff.lower()


# ── Notes ─────────────────────────────────────────────────────────────────────


def test_notes_is_list(two_entries):
    result = compare(two_entries)
    assert isinstance(result.notes, list)


def test_low_confidence_note_present_when_low_entry_exists(two_entries):
    result = compare(two_entries)
    combined = " ".join(result.notes).lower()
    assert "low confidence" in combined


def test_h200_vs_h100_kv_note():
    """H200 vs H100 comparison should produce a KV-budget advantage note."""
    h100_model = get_model("llama-3.1-8b")
    h100_gpu = get_gpu("h100_sxm")
    h200_gpu = get_gpu("h200_sxm")

    base_kwargs = dict(
        requests_per_day=500_000,
        peak_multiplier=2.0,
        isl=4096, osl=256,
        ttft_slo_ms=5000.0,
        model=h100_model,
        dtype="bf16",
        tp=1,
        traffic_class="realtime",
    )
    h100_est = plan(gpu=h100_gpu, **base_kwargs)
    h200_est = plan(gpu=h200_gpu, **base_kwargs)

    entries = [
        ComparisonEntry(label="h100-bf16", estimate=h100_est, gpu_name="h100_sxm",
                        model_name="llama-3.1-8b", dtype="bf16"),
        ComparisonEntry(label="h200-bf16", estimate=h200_est, gpu_name="h200_sxm",
                        model_name="llama-3.1-8b", dtype="bf16"),
    ]
    result = compare(entries)
    combined = " ".join(result.notes).lower()
    assert "h200" in combined
    assert "kv" in combined


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_compare_requires_at_least_two_entries(l4_entry):
    with pytest.raises(ValueError, match="at least 2"):
        compare([l4_entry])


def test_compare_no_cost_falls_back_to_replicas(l4_entry, h100_entry):
    """When no cost is attached, compare by replica count and note missing pricing."""
    l4_no_cost = ComparisonEntry(
        label=l4_entry.label,
        estimate=l4_entry.estimate,
        cost=None,
        gpu_name=l4_entry.gpu_name,
    )
    h100_no_cost = ComparisonEntry(
        label=h100_entry.label,
        estimate=h100_entry.estimate,
        cost=None,
        gpu_name=h100_entry.gpu_name,
    )
    result = compare([l4_no_cost, h100_no_cost])
    assert "no pricing" in result.cheapest.tradeoff.lower() or "catalog" in result.cheapest.tradeoff.lower()
