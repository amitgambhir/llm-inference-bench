"""
planner/cost.py — GPU cost model for a sized capacity estimate.

All costs are derived from modeled throughput (not assumed utilization).
Both on-demand and reserved pricing variants are returned.
The split between prefill and decode cost uses the replicas_prefill :
replicas_decode ratio so the caller can see which workload shape dominates
the bill.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from planner.catalog import CatalogError, _get_catalog
from planner.capacity import CapacityEstimate


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


@dataclass
class CostVariant:
    price_usd_per_gpu_hour: float
    gpu_hours_day: float
    cost_day_usd: float
    cost_month_usd: float
    cost_per_1m_tokens_usd: float
    cost_per_request_usd: float


@dataclass
class CostBreakdown:
    prefill_fraction: float    # fraction of replicas driven by prefill constraint
    decode_fraction: float     # fraction driven by decode constraint
    # cost attributed to each (proportional split of total)
    prefill_cost_day_usd: float
    decode_cost_day_usd: float


@dataclass
class CostEstimate:
    gpu_name: str
    replicas: int
    tp: int
    on_demand: CostVariant
    reserved: CostVariant
    breakdown: CostBreakdown
    # The price source echoed back so reports can show exactly which rate was used
    price_source: str
    warnings: list


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def compute_cost(
    estimate: CapacityEstimate,
    gpu_name: str,
    tp: int = 1,
    override_on_demand_usd_per_hour: Optional[float] = None,
    override_reserved_usd_per_hour: Optional[float] = None,
) -> CostEstimate:
    """Derive cost metrics from a CapacityEstimate.

    Args:
        estimate: output of planner.capacity.plan()
        gpu_name: catalog GPU key (e.g. "h100_sxm")
        tp: tensor parallel degree (each replica uses tp GPUs)
        override_on_demand_usd_per_hour: caller-supplied price; overrides catalog
        override_reserved_usd_per_hour: caller-supplied price; overrides catalog
    """
    warnings: list[str] = []

    # ── Fetch catalog prices ──────────────────────────────────────────────
    cat = _get_catalog()
    cost_profile = cat.costs.get(gpu_name)

    if cost_profile is None:
        if override_on_demand_usd_per_hour is None or override_reserved_usd_per_hour is None:
            available = ", ".join(sorted(cat.costs))
            raise CatalogError(
                f"No cost profile for GPU '{gpu_name}'. Available: {available}. "
                "Add it to catalog/costs.yaml or pass override_on_demand_usd_per_hour "
                "and override_reserved_usd_per_hour."
            )
        warnings.append(
            f"No catalog cost entry for '{gpu_name}'. "
            "Using caller-supplied override prices — verify against your contract."
        )
        on_demand_rate = override_on_demand_usd_per_hour
        reserved_rate = override_reserved_usd_per_hour
        price_source = "caller override"
    else:
        on_demand_rate = override_on_demand_usd_per_hour or cost_profile.on_demand_usd_per_hour
        reserved_rate = override_reserved_usd_per_hour or cost_profile.reserved_usd_per_hour
        price_source = (
            "catalog/costs.yaml"
            if override_on_demand_usd_per_hour is None
            else "catalog default + caller override"
        )

    warnings.append(
        "Cost figures are estimates. Actual pricing depends on provider, region, "
        "commitment tier, and spot availability. Always verify against your contract."
    )

    # ── Core metrics ──────────────────────────────────────────────────────
    replicas = estimate.replicas
    total_tokens_day = estimate.traffic.total_tokens_day
    requests_per_day = estimate.traffic.requests_per_day

    # Each replica occupies tp physical GPUs
    gpu_hours_day = replicas * tp * 24

    def _variant(rate: float) -> CostVariant:
        cost_day = gpu_hours_day * rate
        cost_month = cost_day * 30
        cost_per_1m = cost_day / (total_tokens_day / 1e6) if total_tokens_day > 0 else 0.0
        cost_per_req = cost_day / requests_per_day if requests_per_day > 0 else 0.0
        return CostVariant(
            price_usd_per_gpu_hour=rate,
            gpu_hours_day=gpu_hours_day,
            cost_day_usd=cost_day,
            cost_month_usd=cost_month,
            cost_per_1m_tokens_usd=cost_per_1m,
            cost_per_request_usd=cost_per_req,
        )

    # ── Prefill / decode split ────────────────────────────────────────────
    r_prefill = estimate.replicas_prefill
    r_decode = estimate.replicas_decode
    total_driven = r_prefill + r_decode

    if total_driven > 0:
        prefill_frac = r_prefill / total_driven
        decode_frac = r_decode / total_driven
    else:
        prefill_frac = 0.5
        decode_frac = 0.5

    on_demand = _variant(on_demand_rate)
    reserved = _variant(reserved_rate)

    breakdown = CostBreakdown(
        prefill_fraction=prefill_frac,
        decode_fraction=decode_frac,
        prefill_cost_day_usd=on_demand.cost_day_usd * prefill_frac,
        decode_cost_day_usd=on_demand.cost_day_usd * decode_frac,
    )

    return CostEstimate(
        gpu_name=gpu_name,
        replicas=replicas,
        tp=tp,
        on_demand=on_demand,
        reserved=reserved,
        breakdown=breakdown,
        price_source=price_source,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Human-readable render
# ---------------------------------------------------------------------------


def render_cost(cost: CostEstimate) -> str:
    od = cost.on_demand
    rs = cost.reserved
    bd = cost.breakdown
    lines = [
        "── Cost Estimate ───────────────────────────────────────────",
        f"  GPU            : {cost.gpu_name}  ×{cost.replicas} replicas  (tp={cost.tp})",
        f"  GPU-hours/day  : {od.gpu_hours_day:,.0f}",
        "",
        "  On-demand  (${:.2f}/GPU-hr)".format(od.price_usd_per_gpu_hour),
        f"    /day           : ${od.cost_day_usd:,.2f}",
        f"    /month         : ${od.cost_month_usd:,.2f}",
        f"    /1M tokens     : ${od.cost_per_1m_tokens_usd:.4f}",
        f"    /request       : ${od.cost_per_request_usd:.6f}",
        "",
        "  Reserved   (${:.2f}/GPU-hr)".format(rs.price_usd_per_gpu_hour),
        f"    /day           : ${rs.cost_day_usd:,.2f}",
        f"    /month         : ${rs.cost_month_usd:,.2f}",
        f"    /1M tokens     : ${rs.cost_per_1m_tokens_usd:.4f}",
        f"    /request       : ${rs.cost_per_request_usd:.6f}",
        "",
        "  Cost split (on-demand, by replica driver)",
        f"    Prefill-driven : {bd.prefill_fraction:.0%}  (${bd.prefill_cost_day_usd:,.2f}/day)",
        f"    Decode-driven  : {bd.decode_fraction:.0%}  (${bd.decode_cost_day_usd:,.2f}/day)",
        f"  Price source    : {cost.price_source}",
    ]
    for w in cost.warnings:
        lines.append(f"  ⚠  {w}")
    return "\n".join(lines)
