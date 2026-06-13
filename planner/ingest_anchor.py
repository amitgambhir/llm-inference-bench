"""
planner/ingest_anchor.py — close the loop: read a completed benchmark result,
derive real MFU and bandwidth efficiency, and append a calibrated anchor to
catalog/anchors.yaml.

After ingestion, any capacity.plan() call for a scenario in the same
(model, gpu, dtype, ISL-band) automatically gets higher confidence and uses the
real MFU rather than the GPU default.

Derivation formulas
───────────────────
  measured_prefill_tps   = isl / (ttft_p50_ms / 1000)
  flops_per_token(isl)   = 2 * active_params + 2 * num_layers * isl * d_model
  derived_mfu_prefill    = measured_prefill_tps * flops_per_token / (gpu.peak_flops[dtype] * 1e12)

  # decode: throughput_tokens_per_sec is total output tokens/sec across all concurrent reqs
  batch                  = concurrency
  avg_ctx                = isl + osl // 2
  bytes_per_step         = active_params * weight_bytes + batch * kv_bpt * avg_ctx
  achieved_bw            = throughput_tps * bytes_per_step / batch
  derived_bw_eff_decode  = achieved_bw / (gpu.hbm_bandwidth_gbps * 1e9)

Both values are clamped to [0.01, 0.99] to reject garbage.

CLI:
  python planner/ingest_anchor.py results/real/<tag>.json \\
    --gpu l4 --dtype fp8 [--model llama-3.1-8b] [--osl 128]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml

from planner.catalog import (
    Anchor,
    CatalogError,
    GpuProfile,
    ModelProfile,
    _REPO_CATALOG,
    _get_catalog,
    get_gpu,
    get_model,
    resolve_model,
)
from planner.confidence import compute_confidence_from_anchors

# Path written to by default; tests can redirect with anchors_file parameter
_DEFAULT_ANCHORS_FILE = _REPO_CATALOG / "anchors.yaml"

# Display-name fragments → catalog key (case-insensitive substring match)
_GPU_DISPLAY_TO_KEY: dict[str, str] = {
    "h100": "h100_sxm",
    "h200": "h200_sxm",
    "a100": "a100_80gb_sxm",
    "l40s": "l40s",
    "l4": "l4",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpu_key_from_display(display_name: str) -> Optional[str]:
    """Try to map a GPU display name from result JSON to a catalog key."""
    dn = display_name.lower()
    # Exact match first
    cat = _get_catalog()
    for key, gpu in cat.gpus.items():
        if gpu.display_name.lower() == dn:
            return key
    # Substring heuristics — order matters (l40s before l4)
    for fragment, key in sorted(_GPU_DISPLAY_TO_KEY.items(), key=lambda x: -len(x[0])):
        if fragment in dn:
            return key
    return None


def _clamp(val: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# MFU / bw_eff derivation
# ---------------------------------------------------------------------------


def derive_mfu_prefill(
    isl: int,
    ttft_p50_ms: float,
    model: ModelProfile,
    gpu: GpuProfile,
    dtype: str,
) -> float:
    """Back out realised MFU from measured TTFT p50."""
    measured_prefill_tps = isl / (ttft_p50_ms / 1000.0)
    flops_per_token = (
        2 * model.active_params
        + 2 * model.num_layers * isl * model.d_model
    )
    peak_flops = gpu.peak_flops.get(dtype) * 1e12
    mfu = measured_prefill_tps * flops_per_token / peak_flops
    return _clamp(mfu)


def derive_bw_efficiency_decode(
    throughput_tps: float,
    concurrency: int,
    isl: int,
    osl: int,
    model: ModelProfile,
    gpu: GpuProfile,
) -> float:
    """Back out realised bandwidth efficiency from measured decode throughput."""
    batch = concurrency
    avg_ctx = isl + osl // 2
    bytes_per_step = (
        model.active_params * model.weight_bytes_per_param
        + batch * model.kv_bytes_per_token * avg_ctx
    )
    achieved_bw = throughput_tps * bytes_per_step / max(batch, 1)
    bw_eff = achieved_bw / (gpu.hbm_bandwidth_gbps * 1e9)
    return _clamp(bw_eff)


# ---------------------------------------------------------------------------
# Impact report
# ---------------------------------------------------------------------------


def _confidence_impact_report(
    new_anchor: Anchor,
    model: ModelProfile,
    gpu: GpuProfile,
    dtype: str,
    isl: int,
    osl: int,
    concurrency: int,
    anchors_before: list[Anchor],
    anchors_after: list[Anchor],
) -> list[str]:
    """Return lines describing the before/after confidence change."""
    before = compute_confidence_from_anchors(
        anchors_before, model, gpu, dtype, isl, osl, concurrency
    )
    after = compute_confidence_from_anchors(
        anchors_after, model, gpu, dtype, isl, osl, concurrency
    )

    lines = [
        f"Confidence impact for ({model.name}, {gpu.name}, {dtype}, ISL={isl}):",
        f"  Before ingest : {before.level.upper()}  (band ±{before.band_factor:.0%})",
        f"  After  ingest : {after.level.upper()}  (band ±{after.band_factor:.0%})",
    ]

    if after.level != before.level:
        lines.append(
            f"  → Confidence upgraded {before.level.upper()} → {after.level.upper()}. "
            f"Range narrows from ±{before.band_factor:.0%} to ±{after.band_factor:.0%}."
        )
    else:
        lines.append(f"  → No change in confidence label (already {after.level.upper()}).")

    # Scan for other ISL-band scenarios that would also benefit
    cat = _get_catalog()
    related = [
        a for a in cat.anchors
        if a.model == new_anchor.model
        and a.gpu == new_anchor.gpu
        and a.dtype == new_anchor.dtype
        and a is not new_anchor
    ]
    if related:
        isls = sorted({a.isl for a in related})
        lines.append(
            f"  Related (model, gpu, dtype) anchors exist at ISL={isls}. "
            "Scenarios within ±20% of those ISLs also benefit."
        )

    return lines


# ---------------------------------------------------------------------------
# Core ingest function
# ---------------------------------------------------------------------------


def ingest_anchor(
    result_path: Path,
    gpu_name: str,
    dtype: str,
    model_name: Optional[str] = None,
    osl: Optional[int] = None,
    anchors_file: Optional[Path] = None,
) -> tuple[Anchor, list[str]]:
    """Parse a result JSON, derive calibration coefficients, and append to anchors YAML.

    Returns:
        (new_anchor, impact_lines) where impact_lines is the before/after report.
    """
    anchors_file = anchors_file or _DEFAULT_ANCHORS_FILE

    # ── Parse result JSON ─────────────────────────────────────────────────
    raw = json.loads(result_path.read_text())
    meta = raw.get("meta", {})
    metrics = raw.get("metrics", {})

    resolved_model_name = model_name or meta.get("model")
    if not resolved_model_name:
        raise CatalogError(
            "Cannot determine model name: not in result JSON meta.model and --model not provided."
        )

    isl = int(meta.get("workload", {}).get("isl_approx", 0))
    if isl <= 0:
        raise CatalogError("result JSON missing workload.isl_approx or value is 0.")

    resolved_osl = osl or int(meta.get("workload", {}).get("osl_max", 128))
    concurrency = int(meta.get("workload", {}).get("concurrency", 1))
    tag = meta.get("tag", result_path.stem)

    ttft_p50_ms = float(metrics.get("ttft_ms", {}).get("p50", 0))
    ttft_p95_ms_raw = metrics.get("ttft_ms", {}).get("p95")
    ttft_p95_ms = float(ttft_p95_ms_raw) if ttft_p95_ms_raw is not None else None
    throughput_tps = float(metrics.get("throughput_tokens_per_sec", 0))

    if ttft_p50_ms <= 0:
        raise CatalogError(
            f"Result JSON has invalid ttft_ms.p50={ttft_p50_ms}. Cannot derive MFU."
        )
    if throughput_tps <= 0:
        raise CatalogError(
            f"Result JSON has invalid throughput_tokens_per_sec={throughput_tps}. "
            "Cannot derive bandwidth efficiency."
        )

    # ── Resolve model and GPU profiles ───────────────────────────────────
    try:
        model = get_model(resolved_model_name)
    except CatalogError:
        raise CatalogError(
            f"Model '{resolved_model_name}' not found in catalog. "
            "Add it to catalog/models.yaml or supply --model with a catalog key."
        )

    try:
        gpu = get_gpu(gpu_name)
    except CatalogError:
        raise

    # ── Derive calibration coefficients ──────────────────────────────────
    derived_mfu = derive_mfu_prefill(isl, ttft_p50_ms, model, gpu, dtype)
    derived_bw_eff = derive_bw_efficiency_decode(
        throughput_tps, concurrency, isl, resolved_osl, model, gpu
    )

    # ── Build anchor row ──────────────────────────────────────────────────
    new_anchor = Anchor(
        model=resolved_model_name,
        gpu=gpu_name,
        dtype=dtype,
        isl=isl,
        osl=resolved_osl,
        concurrency=concurrency,
        measured_ttft_p50_ms=ttft_p50_ms,
        measured_ttft_p95_ms=ttft_p95_ms,
        measured_throughput_tok_s=throughput_tps,
        derived_mfu_prefill=derived_mfu,
        source=f"ingest from {tag}",
    )

    # ── Compute before/after confidence impact ────────────────────────────
    anchors_before = _load_anchors(anchors_file)
    anchors_after = anchors_before + [new_anchor]

    impact = _confidence_impact_report(
        new_anchor, model, gpu, dtype, isl, resolved_osl, concurrency,
        anchors_before, anchors_after,
    )

    # ── Append to anchors YAML ────────────────────────────────────────────
    _append_anchor(new_anchor, anchors_file)

    # Invalidate singleton so next catalog load picks up the new anchor
    import planner.catalog as _cm
    _cm._catalog = None

    return new_anchor, impact


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _load_anchors(anchors_file: Path) -> list[Anchor]:
    if not anchors_file.exists():
        return []
    raw = yaml.safe_load(anchors_file.read_text()) or []
    return [Anchor(**row) for row in raw if row is not None]


def _append_anchor(anchor: Anchor, anchors_file: Path) -> None:
    existing_rows: list = []
    if anchors_file.exists():
        existing_rows = yaml.safe_load(anchors_file.read_text()) or []

    # Avoid duplicate: same (model, gpu, dtype, isl, concurrency, source)
    for row in existing_rows:
        if (
            row.get("model") == anchor.model
            and row.get("gpu") == anchor.gpu
            and row.get("dtype") == anchor.dtype
            and row.get("isl") == anchor.isl
            and row.get("concurrency") == anchor.concurrency
            and row.get("source") == anchor.source
        ):
            return  # already present

    row = {
        "model": anchor.model,
        "gpu": anchor.gpu,
        "dtype": anchor.dtype,
        "isl": anchor.isl,
        "osl": anchor.osl,
        "concurrency": anchor.concurrency,
        "measured_ttft_p50_ms": anchor.measured_ttft_p50_ms,
        "measured_ttft_p95_ms": anchor.measured_ttft_p95_ms,
        "measured_throughput_tok_s": anchor.measured_throughput_tok_s,
        "derived_mfu_prefill": anchor.derived_mfu_prefill,
        "source": anchor.source,
    }
    existing_rows.append(row)

    anchors_file.parent.mkdir(parents=True, exist_ok=True)
    with anchors_file.open("w") as f:
        yaml.dump(existing_rows, f, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest a benchmark result into the anchor catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("result", help="Path to results/real/<tag>.json")
    p.add_argument("--gpu", required=True, help="Catalog GPU key (e.g. l4, h100_sxm)")
    p.add_argument("--dtype", required=True,
                   choices=["fp32", "bf16", "fp16", "fp8", "mxfp4", "int8", "int4"],
                   help="Serving dtype used during the benchmark run")
    p.add_argument("--model", default=None,
                   help="Catalog model key (default: read from result meta.model)")
    p.add_argument("--osl", type=int, default=None,
                   help="Output sequence length (default: read from result meta.workload.osl_max)")
    p.add_argument("--anchors-file", default=None,
                   help="Override path to anchors YAML (default: catalog/anchors.yaml)")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    result_path = Path(args.result)
    if not result_path.exists():
        print(f"Error: result file not found: {result_path}", file=sys.stderr)
        sys.exit(1)

    anchors_file = Path(args.anchors_file) if args.anchors_file else None

    try:
        anchor, impact = ingest_anchor(
            result_path=result_path,
            gpu_name=args.gpu,
            dtype=args.dtype,
            model_name=args.model,
            osl=args.osl,
            anchors_file=anchors_file,
        )
    except CatalogError as e:
        print(f"Ingest error: {e}", file=sys.stderr)
        sys.exit(1)

    target = anchors_file or _DEFAULT_ANCHORS_FILE
    print(f"Anchor appended to {target}")
    print(f"  model={anchor.model}  gpu={anchor.gpu}  dtype={anchor.dtype}")
    print(f"  isl={anchor.isl}  osl={anchor.osl}  concurrency={anchor.concurrency}")
    print(f"  derived_mfu_prefill={anchor.derived_mfu_prefill:.4f}")
    print(f"  measured_ttft_p50_ms={anchor.measured_ttft_p50_ms}")
    print(f"  measured_throughput_tok_s={anchor.measured_throughput_tok_s}")
    print()
    for line in impact:
        print(line)


if __name__ == "__main__":
    main()
