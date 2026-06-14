"""
planner/capacity.py — anchored roofline LLM inference capacity planner.

Separates prefill (compute-bound) from decode (bandwidth-bound), caps
concurrency via KV-cache budget, and labels confidence from anchor
availability.  Never emits a bare number — every result carries
{value, range, binding_constraint, confidence, assumptions}.

CLI:
  python planner/capacity.py \\
    --requests-per-day 100000000 --peak-multiplier 3 \\
    --isl 9000 --osl 500 --ttft-slo-ms 2000 \\
    --model gpt-oss-20b --gpu h100_sxm \\
    --dtype mxfp4 --runtime vllm --tp 1 --traffic-class realtime \\
    [--json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from planner.catalog import (
    DTYPE_BYTES,
    CatalogError,
    GpuProfile,
    ModelProfile,
    get_gpu,
    get_model,
    resolve_gpu,
    resolve_model,
)
# confidence and ConfidenceResult live in planner.confidence; re-exported here
# so existing imports from planner.capacity continue to work.
from planner.confidence import ConfidenceResult, confidence  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXED_OVERHEAD_BYTES = 0.5 * 1e9        # framework buffers; CUDA context lives in the gpu_mem_util margin
LOW_KV_CONCURRENCY_THRESHOLD = 4        # warn when max_concurrent_seqs falls below this
CHUNKED_PREFILL_ISL_THRESHOLD = 4096    # vLLM needs chunked prefill above this
MAX_QUEUE_UTILIZATION = 0.94            # clamp rho below this to avoid M/M/1 blow-up

HEADROOM_FACTORS = {
    "realtime": 1.40,
    "mixed": 1.25,
    "batch": 1.10,
}

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Traffic:
    requests_per_day: int
    avg_rps: float
    peak_rps: float
    input_tps_avg: float
    output_tps_avg: float
    input_tps_peak: float
    output_tps_peak: float
    total_tokens_day: float


@dataclass
class KvBudget:
    kv_bytes_per_token: float
    weights_resident_bytes: float
    usable_mem_bytes: float
    kv_cache_budget_bytes: float
    max_kv_tokens: float
    max_concurrent_seqs: int


@dataclass
class TtftEstimate:
    ttft_compute_ms: float
    ttft_queue_ms: float
    ttft_ms: float
    utilization: float
    slo_ms: float
    slo_met: bool
    slo_breach_reason: Optional[str]


@dataclass
class CapacityEstimate:
    # Core traffic and memory
    traffic: Traffic
    kv_budget: KvBudget
    # Per-GPU ceilings
    prefill_tps_gpu: float
    decode_tps_gpu: float
    # Sizing — top-level fields per plan spec
    replicas: int
    replicas_low: int
    replicas_high: int
    binding_constraint: str     # "prefill-bound" | "decode-bound" | "kv-memory-bound"
    ttft_estimate: TtftEstimate
    confidence: str             # "high" | "medium" | "low"
    assumptions: list
    warnings: list
    # Detailed breakdown (not in plan spec but useful for debugging)
    replicas_prefill: int = 0
    replicas_decode: int = 0
    replicas_concurrency: int = 0
    mfu_used: float = 0.0
    bw_eff_used: float = 0.0
    decode_bw_eff_used: float = 0.0   # after KV-ratio adjustment
    anchor_matched: bool = False
    tp_used: int = 1                  # tensor parallel degree used
    # Throughput-centric metrics
    total_gpus: int = 0               # replicas × tp (true fleet size)
    tpot_ms: float = 0.0              # decode inter-token latency per request at eff_batch
    eff_batch_used: int = 0           # batch_efficiency × max_concurrent_seqs
    kv_ratio: float = 0.0             # kv_bytes / weight_bytes at eff_batch (decode regime indicator)


# ---------------------------------------------------------------------------
# Step 1: normalize_traffic
# ---------------------------------------------------------------------------


def normalize_traffic(
    requests_per_day: int,
    peak_multiplier: float,
    isl: int,
    osl: int,
) -> Traffic:
    avg_rps = requests_per_day / 86_400
    peak_rps = avg_rps * peak_multiplier
    return Traffic(
        requests_per_day=requests_per_day,
        avg_rps=avg_rps,
        peak_rps=peak_rps,
        input_tps_avg=avg_rps * isl,
        output_tps_avg=avg_rps * osl,
        input_tps_peak=peak_rps * isl,
        output_tps_peak=peak_rps * osl,
        total_tokens_day=requests_per_day * (isl + osl),
    )


# ---------------------------------------------------------------------------
# Step 2: kv_budget
# ---------------------------------------------------------------------------


def kv_budget(
    gpu: GpuProfile,
    model: ModelProfile,
    dtype: str,
    isl: int,
    osl: int,
    gpu_mem_util: float = 0.90,
    tp: int = 1,
) -> KvBudget:
    kv_bpt = model.kv_bytes_per_token
    # Use resident_weights_gb when available — accounts for mixed-precision checkpoints
    # (e.g. MoE with 4-bit experts + bf16 attention/norm layers) where a flat dtype
    # multiplied by total_params materially underestimates real memory footprint.
    if model.resident_weights_gb is not None:
        weights_bytes = model.resident_weights_gb * 1e9 / tp
    else:
        weights_bytes = model.total_params * DTYPE_BYTES.get(dtype, 2.0) / tp
    usable_mem = gpu.mem_gb * 1e9 * gpu_mem_util

    if weights_bytes > usable_mem:
        raise CatalogError(
            f"Model '{model.name}' requires {weights_bytes/1e9:.1f} GB per GPU "
            f"for weights at {dtype} (tp={tp}) but GPU '{gpu.name}' has only "
            f"{usable_mem/1e9:.1f} GB usable (mem_gb={gpu.mem_gb}, "
            f"gpu_mem_util={gpu_mem_util}). "
            "Use a larger tp or a GPU with more VRAM."
        )

    # KV cache shards across min(tp, num_kv_heads) ranks.  When tp > num_kv_heads
    # (e.g. tp=8, Qwen3-30B with 4 KV heads) excess ranks replicate KV rather than
    # extending the budget — capping kv_shard_factor prevents over-counting.
    kv_shard_factor = min(tp, model.num_kv_heads)
    kv_cache_budget = (usable_mem - weights_bytes - FIXED_OVERHEAD_BYTES) * kv_shard_factor
    if kv_cache_budget <= 0:
        raise CatalogError(
            f"No KV cache budget after model weights + fixed overhead for "
            f"'{model.name}' on '{gpu.name}' (tp={tp})."
        )

    max_kv_tokens = kv_cache_budget / kv_bpt
    max_concurrent_seqs = max(1, math.floor(max_kv_tokens / (isl + osl)))

    return KvBudget(
        kv_bytes_per_token=kv_bpt,
        weights_resident_bytes=weights_bytes,
        usable_mem_bytes=usable_mem,
        kv_cache_budget_bytes=kv_cache_budget,
        max_kv_tokens=max_kv_tokens,
        max_concurrent_seqs=max_concurrent_seqs,
    )


# ---------------------------------------------------------------------------
# Step 3: prefill_ceiling
# ---------------------------------------------------------------------------


def prefill_ceiling(
    gpu: GpuProfile,
    model: ModelProfile,
    dtype: str,
    isl: int,
    mfu: float,
    bw_eff: float = 0.70,
    tp: int = 1,
) -> float:
    """Return prefill throughput in tokens/sec for the TP group — min(compute, bandwidth).

    FLOPs per token = 2 * active_params (linear layers) +
                      2 * num_layers * isl * d_model (attention — grows with context).

    For most throughput workloads (ISL ≥ ~200 on H100) prefill is compute-bound and
    the bandwidth floor does not bind.  It matters for short-prompt workloads (ISL < 170)
    where a single request's tokens cannot amortise the full weight-read over enough
    compute to saturate the tensor cores.

    Ridge point (tokens): peak_flops * mfu / (bandwidth * bw_eff)
      ≈ 989e12 * 0.40 / (3350e9 * 0.70) ≈ 169 tokens on H100 bf16

    Both ceilings include the TP multiplier (TP group has N× compute and N× bandwidth).
    """
    # Attention O(n²d) term uses the Q-projection dimension, not d_model.
    # For standard models num_q_heads × head_dim == d_model, but for GQA-heavy
    # or atypical architectures (e.g. Qwen3: d_model=2048, Q-dim=4096) they differ.
    q_proj_dim = model.num_q_heads * model.head_dim
    flops_per_token = (
        2 * model.active_params
        + 2 * model.num_layers * isl * q_proj_dim
    )
    # Compute-bound: FLOPs limited (TP group has tp × peak_flops)
    compute_tps = gpu.peak_flops.get(dtype) * 1e12 * mfu * tp / flops_per_token

    # Bandwidth-bound: weight-streaming limited when ISL < ridge point.
    # Processing isl tokens loads weights once; throughput = isl / weight_load_time.
    weight_bytes = model.active_params * DTYPE_BYTES.get(dtype, 2.0)
    bw_tps = isl * gpu.hbm_bandwidth_gbps * 1e9 * bw_eff * tp / weight_bytes

    return min(compute_tps, bw_tps)


# ---------------------------------------------------------------------------
# Step 4: decode_ceiling
# ---------------------------------------------------------------------------


def decode_ceiling(
    gpu: GpuProfile,
    model: ModelProfile,
    dtype: str,
    batch: int,
    avg_ctx: int,
    bw_eff: float,
    tp: int = 1,
    mfu: Optional[float] = None,
) -> float:
    """Return decode throughput in tokens/sec for the TP group at the given batch + bw_eff.

    Weight reads amortize over the batch; KV reads do NOT (each seq reads its own cache).
    With TP=N each GPU reads 1/N of weights and 1/N of KV, but N GPUs run in
    parallel — net effect: throughput scales linearly with tp.

    MoE expert coverage: at large batch, tokens scatter to different experts and the
    union of experts streamed per step approaches total expert params.  Uses a
    birthday-problem fraction: distinct_frac(B) = 1 − (1 − k/n)^B.

    Compute ceiling: rarely binds (decode is almost always bandwidth-bound), but
    applies at very large batch on small models where GEMM FLOPs saturate before HBM.
    """
    achievable_bw = gpu.hbm_bandwidth_gbps * 1e9 * bw_eff
    dtype_bytes = DTYPE_BYTES.get(dtype, 2.0)

    # Decode weight bytes per step: for MoE, different sequences route to different
    # experts.  At batch B the expected fraction of the expert pool streamed is
    # 1 − (1 − experts_per_token/num_experts)^B (birthday problem).
    if model.is_moe and model.num_experts and model.experts_per_token:
        r = model.experts_per_token / model.num_experts
        distinct_frac = 1.0 - (1.0 - r) ** batch
        # Derive dense (non-expert) params: solve active = dense + r × total
        dense_params = (model.active_params - r * model.total_params) / (1.0 - r)
        expert_pool_params = model.total_params - dense_params
        weight_bytes_step = (dense_params + distinct_frac * expert_pool_params) * dtype_bytes
    else:
        weight_bytes_step = model.active_params * dtype_bytes

    bytes_per_step = (
        weight_bytes_step / tp
        + batch * model.kv_bytes_per_token * avg_ctx / tp
    )
    bw_tps = batch * achievable_bw / bytes_per_step

    # Compute ceiling: FLOPs/step = 2 × active_params × batch; batch cancels in TPS.
    _mfu = mfu if mfu is not None else gpu.default_mfu_prefill
    compute_tps = gpu.peak_flops.get(dtype) * 1e12 * _mfu * tp / (2 * model.active_params)

    return min(bw_tps, compute_tps)


# ---------------------------------------------------------------------------
# Step 5: confidence — defined in planner.confidence, re-exported above.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 6: ttft_estimate
# ---------------------------------------------------------------------------


def ttft_estimate(
    gpu: GpuProfile,
    model: ModelProfile,
    dtype: str,
    isl: int,
    prefill_tps_per_replica: float,
    utilization: float,
    ttft_slo_ms: float,
) -> TtftEstimate:
    """Estimate TTFT for a single replica at the given utilization.

    Uses a simple M/M/1-style queuing heuristic — suitable for order-of-magnitude
    planning, but must be validated on a live GPU before capacity commitment.

    ttft_compute = isl / prefill_tps_per_replica
    queue_wait   = ttft_compute * rho / (1 - rho)   [M/M/1 mean wait]
    """
    rho = min(utilization, MAX_QUEUE_UTILIZATION)
    ttft_compute_s = isl / prefill_tps_per_replica
    queue_wait_s = ttft_compute_s * (rho / (1.0 - rho)) if rho > 0 else 0.0
    ttft_ms = (ttft_compute_s + queue_wait_s) * 1000.0

    slo_met = ttft_ms <= ttft_slo_ms
    breach_reason: Optional[str] = None
    if not slo_met:
        if queue_wait_s > ttft_compute_s:
            breach_reason = (
                f"Queue-bound: queuing delay ({queue_wait_s*1000:.0f} ms) exceeds "
                f"compute time ({ttft_compute_s*1000:.0f} ms). "
                "Add replicas or reduce peak load."
            )
        else:
            breach_reason = (
                f"Compute-bound: prefill of {isl} tokens takes "
                f"{ttft_compute_s*1000:.0f} ms, exceeding SLO of {ttft_slo_ms:.0f} ms. "
                "Use a faster GPU, higher MFU (chunked prefill), or reduce ISL."
            )

    return TtftEstimate(
        ttft_compute_ms=ttft_compute_s * 1000.0,
        ttft_queue_ms=queue_wait_s * 1000.0,
        ttft_ms=ttft_ms,
        utilization=rho,
        slo_ms=ttft_slo_ms,
        slo_met=slo_met,
        slo_breach_reason=breach_reason,
    )


# ---------------------------------------------------------------------------
# Step 7: size_replicas
# ---------------------------------------------------------------------------


def size_replicas(
    traffic: Traffic,
    prefill_tps_gpu: float,
    decode_tps_gpu: float,
    max_concurrent_seqs: int,
    tp: int,
    traffic_class: str,
    isl: int,
    osl: int,
    eff_batch: Optional[int] = None,
) -> dict:
    """Compute replica count from three independent constraints.

    Returns a dict with keys:
      replicas_prefill, replicas_decode, replicas_concurrency,
      base_replicas, binding_constraint, headroom_factor, replicas,
      replicas_low, replicas_high  (range widened separately by caller)
    """
    # prefill_tps_gpu and decode_tps_gpu are TP-group throughput (from ceiling functions).
    # A "replica" = one TP group.  Do NOT multiply by tp again.
    per_replica_prefill = prefill_tps_gpu
    per_replica_decode = decode_tps_gpu

    replicas_prefill = math.ceil(traffic.input_tps_peak / per_replica_prefill)
    replicas_decode = math.ceil(traffic.output_tps_peak / per_replica_decode)

    # Little's Law: E[concurrent requests] = λ * E[latency]
    # Rough latency: compute-only TTFT (no queue) + osl * ITL
    # ITL = step_time at the effective batch (consistent with how decode_tps_gpu was computed).
    # Using max_concurrent_seqs here would inflate ITL by 1/batch_efficiency ≈ 1.43×.
    _eff_batch = eff_batch if eff_batch is not None else max(1, int(max_concurrent_seqs * 0.70))
    ttft_rough_s = isl / per_replica_prefill
    itl_s = _eff_batch / decode_tps_gpu
    avg_latency_s = ttft_rough_s + osl * itl_s
    peak_concurrent_req = traffic.peak_rps * avg_latency_s
    replicas_concurrency = math.ceil(peak_concurrent_req / max_concurrent_seqs)

    base_replicas = max(replicas_prefill, replicas_decode, replicas_concurrency)
    if base_replicas == replicas_prefill:
        binding_constraint = "prefill-bound"
    elif base_replicas == replicas_decode:
        binding_constraint = "decode-bound"
    else:
        binding_constraint = "kv-memory-bound"

    headroom = HEADROOM_FACTORS.get(traffic_class, 1.25)
    replicas = math.ceil(base_replicas * headroom)

    return dict(
        replicas_prefill=replicas_prefill,
        replicas_decode=replicas_decode,
        replicas_concurrency=replicas_concurrency,
        base_replicas=base_replicas,
        binding_constraint=binding_constraint,
        headroom_factor=headroom,
        replicas=replicas,
    )


# ---------------------------------------------------------------------------
# Main entry point: plan()
# ---------------------------------------------------------------------------


def plan(
    requests_per_day: int,
    peak_multiplier: float,
    isl: int,
    osl: int,
    ttft_slo_ms: float,
    model: ModelProfile,
    gpu: GpuProfile,
    dtype: str,
    tp: int,
    traffic_class: str,
    gpu_mem_util: float = 0.90,
    runtime: str = "vllm",
    batch_efficiency: float = 0.70,
) -> CapacityEstimate:
    """Full planning pipeline.  Returns a CapacityEstimate with range + confidence.

    batch_efficiency: fraction of max_concurrent_seqs that are actually in flight
        in steady-state continuous batching (vLLM default ≈ 0.70).  The roofline
        decode ceiling is computed at this effective batch rather than the
        theoretical maximum — prevents the common under-provisioning mistake.
    """

    warnings: list[str] = []
    assumptions: list[str] = []

    # ── 0. Upfront validation ─────────────────────────────────────────────
    if dtype not in DTYPE_BYTES:
        raise CatalogError(
            f"Unknown dtype '{dtype}'. Supported: {list(DTYPE_BYTES.keys())}"
        )
    gpu.peak_flops.get(dtype)   # raises CatalogError early if GPU doesn't support this dtype

    # ── 1. Traffic ────────────────────────────────────────────────────────
    traffic = normalize_traffic(requests_per_day, peak_multiplier, isl, osl)

    # ── 2. KV budget ──────────────────────────────────────────────────────
    kv = kv_budget(gpu, model, dtype, isl, osl, gpu_mem_util, tp)

    # ── 3. Confidence + calibrated MFU / bw_eff ───────────────────────────
    conf = confidence(model, gpu, dtype, isl, osl, kv.max_concurrent_seqs)
    mfu = conf.mfu_used
    bw_eff = conf.bw_eff_used

    # ── 4. Ceilings ───────────────────────────────────────────────────────
    pfill_tps = prefill_ceiling(gpu, model, dtype, isl, mfu, bw_eff, tp)

    avg_ctx = isl + osl // 2        # average context mid-generation

    # Effective batch: continuous batching is never 100% full in steady state.
    eff_batch = max(1, int(kv.max_concurrent_seqs * batch_efficiency))

    # KV-ratio-aware bandwidth efficiency for decode.
    # When KV traffic >> weight traffic, scattered PagedAttention reads degrade
    # effective HBM bandwidth below the flat bw_eff assumption.
    #   kv_ratio = total_kv_bytes_in_flight / weight_bytes
    weight_bytes_active = model.active_params * DTYPE_BYTES.get(dtype, 2.0)
    kv_bytes_inflight = kv.kv_bytes_per_token * avg_ctx * eff_batch
    kv_ratio = kv_bytes_inflight / max(weight_bytes_active, 1.0)
    if kv_ratio > 10.0:
        decode_bw_eff = max(0.22, bw_eff * 0.70)   # very KV-heavy (long context + large batch)
    elif kv_ratio > 3.0:
        decode_bw_eff = max(0.22, bw_eff * 0.85)   # moderately KV-heavy
    else:
        decode_bw_eff = bw_eff                       # weight-dominant (short context or small batch)

    decode_tps = decode_ceiling(gpu, model, dtype, eff_batch, avg_ctx, decode_bw_eff, tp, mfu=mfu)

    # ── 5. Sizing ─────────────────────────────────────────────────────────
    sz = size_replicas(traffic, pfill_tps, decode_tps, kv.max_concurrent_seqs, tp, traffic_class, isl, osl, eff_batch=eff_batch)

    # ── 6. Range widening by confidence ───────────────────────────────────
    band = conf.band_factor
    replicas_low = max(1, math.ceil(sz["replicas"] * (1.0 - band)))
    replicas_high = math.ceil(sz["replicas"] * (1.0 + band))

    # ── 7. TTFT at sized replica count ────────────────────────────────────
    per_replica_prefill = pfill_tps  # already TP-group throughput from prefill_ceiling
    # Utilization at peak with the sized fleet
    rho = (traffic.input_tps_peak / (sz["replicas"] * per_replica_prefill))
    ttft = ttft_estimate(gpu, model, dtype, isl, per_replica_prefill, rho, ttft_slo_ms)

    # ── 8. Warnings ───────────────────────────────────────────────────────
    if kv.max_concurrent_seqs < LOW_KV_CONCURRENCY_THRESHOLD:
        warnings.append(
            f"Very limited KV cache: only {kv.max_concurrent_seqs} concurrent sequence(s) fit "
            f"per replica at {dtype} on {gpu.name}. "
            f"Consider increasing --tp (currently {tp}) or using a GPU with more VRAM."
        )
    # Balanced prefill/decode warning.
    # size_replicas uses max(replicas_prefill, replicas_decode), which assumes the
    # binding workload owns the full GPU.  For colocated vLLM this is optimistic when
    # both phases are significant: prefill stresses compute; decode stresses bandwidth;
    # they share the same GPU and can contend.  When neither dominates by ≥2×, flag it.
    pf_r = sz["replicas_prefill"]
    dc_r = sz["replicas_decode"]
    if pf_r > 0 and dc_r > 0:
        _dom = max(pf_r, dc_r)
        _sub = min(pf_r, dc_r)
        if _dom < _sub * 2:
            warnings.append(
                f"Balanced prefill/decode load: prefill-driven={pf_r} replicas, "
                f"decode-driven={dc_r} replicas (ratio {_dom/_sub:.1f}×). "
                "Sizing uses max() which assumes independent resource pools. "
                "For colocated vLLM both phases share the GPU — actual requirement "
                "may be higher. Validate with run_bench.py before committing."
            )

    if conf.level == "low":
        warnings.append(
            "LOW confidence: estimate uses GPU-default MFU and bandwidth efficiency. "
            "Validate on a live GPU before committing infrastructure."
        )
    if isl >= CHUNKED_PREFILL_ISL_THRESHOLD:
        warnings.append(
            f"ISL {isl} ≥ {CHUNKED_PREFILL_ISL_THRESHOLD}: chunked prefill required "
            f"(--enable-chunked-prefill for vLLM). Without it, TTFT will be much worse "
            f"at high concurrency."
        )
    if model.geometry_source == "estimated":
        warnings.append(
            "Model geometry estimated from param count, not from a model card. "
            "Verify num_layers, d_model, and head configuration before production use."
        )
    if not ttft.slo_met and ttft.slo_breach_reason:
        warnings.append(f"TTFT SLO BREACH: {ttft.slo_breach_reason}")
    for note in conf.notes:
        warnings.append(note)

    # ── 9. Assumptions ────────────────────────────────────────────────────
    kv_regime = (
        "very KV-heavy" if kv_ratio > 10.0 else
        "KV-heavy" if kv_ratio > 3.0 else
        "weight-dominant"
    )
    assumptions += [
        f"GPU memory utilization cap: {gpu_mem_util:.0%}",
        f"MFU (prefill): {mfu:.2f} ({'from anchor' if conf.anchor_matched else 'GPU default'})",
        f"Bandwidth efficiency (decode): {decode_bw_eff:.2f} (base={bw_eff:.2f}, KV ratio={kv_ratio:.1f} → {kv_regime})",
        f"Batch efficiency: {batch_efficiency:.0%} of max_concurrent_seqs ({eff_batch}/{kv.max_concurrent_seqs} seqs)",
        f"Traffic headroom ({traffic_class}): {sz['headroom_factor']:.2f}x",
        f"Tensor parallelism: tp={tp} ({sz['replicas']} replicas × {tp} GPUs = {sz['replicas'] * tp} total GPUs)",
        f"Fixed overhead per GPU: {FIXED_OVERHEAD_BYTES/1e9:.0f} GB",
        f"Average context for decode sizing: {isl + osl // 2} tokens (ISL + OSL/2)",
        "TTFT queue model: M/M/1 heuristic — must be validated on live GPU.",
    ]

    return CapacityEstimate(
        traffic=traffic,
        kv_budget=kv,
        prefill_tps_gpu=pfill_tps,
        decode_tps_gpu=decode_tps,
        replicas=sz["replicas"],
        replicas_low=replicas_low,
        replicas_high=replicas_high,
        binding_constraint=sz["binding_constraint"],
        ttft_estimate=ttft,
        confidence=conf.level,
        assumptions=assumptions,
        warnings=warnings,
        replicas_prefill=sz["replicas_prefill"],
        replicas_decode=sz["replicas_decode"],
        replicas_concurrency=sz["replicas_concurrency"],
        mfu_used=mfu,
        bw_eff_used=bw_eff,
        decode_bw_eff_used=decode_bw_eff,
        anchor_matched=conf.anchor_matched,
        tp_used=tp,
        total_gpus=sz["replicas"] * tp,
        tpot_ms=(eff_batch / decode_tps) * 1000.0 if decode_tps > 0 else 0.0,
        eff_batch_used=eff_batch,
        kv_ratio=kv_ratio,
    )


# ---------------------------------------------------------------------------
# Human-readable text output
# ---------------------------------------------------------------------------


def _fmt_num(n: float, suffix: str = "") -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B{suffix}"
    if n >= 1e6:
        return f"{n/1e6:.2f}M{suffix}"
    if n >= 1e3:
        return f"{n/1e3:.2f}K{suffix}"
    return f"{n:.2f}{suffix}"


def render(est: CapacityEstimate, model: ModelProfile, gpu: GpuProfile) -> str:
    t = est.traffic
    kv = est.kv_budget
    tf = est.ttft_estimate
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║           LLM Inference Capacity Estimate                ║",
        "╚══════════════════════════════════════════════════════════╝",
        f"  Model : {model.display_name}  |  GPU: {gpu.display_name}  |  tp={est.tp_used}",
        "",
        "── Traffic ─────────────────────────────────────────────────",
        f"  Avg RPS       : {t.avg_rps:,.1f}",
        f"  Peak RPS      : {t.peak_rps:,.1f}",
        f"  Input TPS peak: {_fmt_num(t.input_tps_peak)}",
        f"  Output TPS pk : {_fmt_num(t.output_tps_peak)}",
        "",
        "── KV Budget (per GPU) ─────────────────────────────────────",
        f"  Weights       : {kv.weights_resident_bytes/1e9:.1f} GB",
        f"  KV cache      : {kv.kv_cache_budget_bytes/1e9:.1f} GB",
        f"  Max seqs/GPU  : {kv.max_concurrent_seqs}",
        "",
        "── Ceilings (per TP group) ─────────────────────────────────",
        f"  Prefill TPS   : {_fmt_num(est.prefill_tps_gpu)}  (MFU={est.mfu_used:.0%})",
        f"  Decode TPS    : {_fmt_num(est.decode_tps_gpu)}  (bw_eff={est.decode_bw_eff_used:.0%}, KV ratio={est.kv_ratio:.1f})",
        f"  Eff batch     : {est.eff_batch_used} seqs  (batch_efficiency × max_seqs)",
        f"  TPOT          : {est.tpot_ms:.1f} ms/token  (at eff_batch)",
        "",
        "── Sizing ──────────────────────────────────────────────────",
        f"  Replicas (prefill-driven) : {est.replicas_prefill}",
        f"  Replicas (decode-driven)  : {est.replicas_decode}",
        f"  Replicas (concurrency)    : {est.replicas_concurrency}",
        f"  Binding constraint        : {est.binding_constraint.upper()}",
        f"  Base replicas             : {max(est.replicas_prefill, est.replicas_decode, est.replicas_concurrency)}",
        f"  After headroom            : {est.replicas}",
        f"  Total GPUs                : {est.total_gpus}  ({est.replicas} replicas × tp)",
        "",
        f"  ┌─────────────────────────────────────────────────┐",
        f"  │  RECOMMENDED:  {est.replicas_low} – {est.replicas_high} replicas            │",
        f"  │  Confidence  :  {est.confidence.upper():<6}                          │",
        f"  └─────────────────────────────────────────────────┘",
        "",
        "── TTFT Estimate ───────────────────────────────────────────",
        f"  Compute       : {tf.ttft_compute_ms:.0f} ms",
        f"  Queue wait    : {tf.ttft_queue_ms:.0f} ms  (utilization={tf.utilization:.0%})",
        f"  Total est.    : {tf.ttft_ms:.0f} ms  {'✓ SLO met' if tf.slo_met else '✗ SLO BREACH'}  (SLO={tf.slo_ms:.0f} ms)",
        "",
    ]
    if est.warnings:
        lines.append("── Warnings ─────────────────────────────────────────────")
        for w in est.warnings:
            lines.append(f"  ⚠  {w}")
        lines.append("")
    lines.append("── Assumptions ─────────────────────────────────────────────")
    for a in est.assumptions:
        lines.append(f"  •  {a}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM Inference Capacity Planner — roofline sizing with confidence labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--requests-per-day", type=float, required=True)
    p.add_argument("--peak-multiplier", type=float, default=3.0)
    p.add_argument("--isl", type=int, required=True, help="Input sequence length (tokens)")
    p.add_argument("--osl", type=int, required=True, help="Output sequence length (tokens)")
    p.add_argument("--ttft-slo-ms", type=float, default=2000.0)

    # Model: catalog name, full spec file, or rough param-only spec
    mg = p.add_mutually_exclusive_group(required=True)
    mg.add_argument("--model", help="Catalog model name (e.g. gpt-oss-20b)")
    mg.add_argument("--model-spec", metavar="FILE", help="Path to full model spec YAML")
    mg.add_argument("--model-params", type=float, metavar="PARAMS",
                    help="Total param count for rough spec (e.g. 20e9)")

    p.add_argument("--model-active", type=float, metavar="PARAMS",
                   help="Active params for MoE rough spec (default=total)")
    p.add_argument("--model-name", default="custom-model",
                   help="Name for rough-spec model (default: custom-model)")

    # GPU: catalog name or spec file
    p.add_argument("--gpu", default="h100_sxm", help="Catalog GPU name (default: h100_sxm)")
    p.add_argument("--gpu-spec", metavar="FILE", help="Path to GPU spec YAML (overrides --gpu)")

    p.add_argument("--dtype", default="bf16",
                   choices=["fp32", "bf16", "fp16", "fp8", "mxfp4", "int8", "int4"])
    p.add_argument("--runtime", default="vllm", choices=["vllm", "sglang", "managed"])
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel degree")
    p.add_argument("--traffic-class", default="realtime",
                   choices=["realtime", "mixed", "batch"])
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human output")
    return p


def _resolve_model_cli(args: argparse.Namespace) -> ModelProfile:
    if args.model:
        return resolve_model(args.model)
    if args.model_spec:
        raw = yaml.safe_load(Path(args.model_spec).read_text())
        return resolve_model(raw)
    # Rough spec from params
    spec: dict = {
        "name": args.model_name,
        "total_params": int(args.model_params),
        "native_dtype": args.dtype,
    }
    if args.model_active:
        spec["active_params"] = int(args.model_active)
    return resolve_model(spec)


def _resolve_gpu_cli(args: argparse.Namespace) -> GpuProfile:
    if args.gpu_spec:
        raw = yaml.safe_load(Path(args.gpu_spec).read_text())
        return resolve_gpu(raw)
    return resolve_gpu(args.gpu)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        model = _resolve_model_cli(args)
        gpu = _resolve_gpu_cli(args)
    except (CatalogError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        est = plan(
            requests_per_day=int(args.requests_per_day),
            peak_multiplier=args.peak_multiplier,
            isl=args.isl,
            osl=args.osl,
            ttft_slo_ms=args.ttft_slo_ms,
            model=model,
            gpu=gpu,
            dtype=args.dtype,
            tp=args.tp,
            traffic_class=args.traffic_class,
            gpu_mem_util=args.gpu_mem_util,
            runtime=args.runtime,
        )
    except CatalogError as e:
        print(f"Planning error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(asdict(est), indent=2, default=str))
    else:
        print(render(est, model, gpu))


if __name__ == "__main__":
    main()
