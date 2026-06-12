#!/usr/bin/env python3
"""
Quality-aware deployment advisor.

Merges latency benchmark results with quality evaluation sidecars to
produce a deployment recommendation balancing latency, cost, and quality.
"""
import json
import os
import sys


REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
REAL_DIR = os.path.join(REPO_ROOT, "results", "real")
SYN_DIR = os.path.join(REPO_ROOT, "results", "synthetic")
QUALITY_DIR = os.path.join(REPO_ROOT, "results", "quality")


def _find_latency_file(tag, latency_dirs):
    """Return path to latency JSON for tag. Later dirs override earlier ones."""
    found = None
    for d in latency_dirs:
        path = os.path.join(d, tag + ".json")
        if os.path.isfile(path):
            found = path
    return found


def validate_profile(profile):
    """Fail fast if required latency fields are missing or None."""
    required = {"ttft_ms_p50", "ttft_ms_p95", "throughput_tokens_per_sec"}
    lat = profile.get("latency", {})
    missing = {f for f in required if lat.get(f) is None}
    if missing:
        print(
            "ERROR: profile '{}' missing latency fields: {}".format(
                profile.get("tag"), missing
            ),
            file=sys.stderr,
        )
        sys.exit(1)


def load_deployment(tag, latency_dirs, quality_dir):
    """
    Load and merge latency + quality data into a normalized DeploymentProfile.

    Flattens the existing nested result schema (e.g. metrics.ttft_ms.p50)
    into a flat in-memory shape (latency.ttft_ms_p50) so all downstream
    functions work against a single consistent structure.
    """
    lat_path = _find_latency_file(tag, latency_dirs)
    if lat_path is None:
        print("ERROR: no latency result found for tag '{}'".format(tag), file=sys.stderr)
        sys.exit(1)

    with open(lat_path) as f:
        lat_raw = json.load(f)

    meta = lat_raw.get("meta", {})
    m = lat_raw.get("metrics", {})
    ttft = m.get("ttft_ms", {})

    profile = {
        "tag": tag,
        "model": meta.get("model", "unknown"),
        "latency": {
            "ttft_ms_p50": ttft.get("p50"),
            "ttft_ms_p95": ttft.get("p95"),
            "throughput_tokens_per_sec": m.get("throughput_tokens_per_sec"),
        },
        "quality": None,
        "cost": {
            "per_million_tokens": None,
            "throughput_proxy_tokens_per_sec": m.get("throughput_tokens_per_sec"),
        },
        "_dataset": None,
    }
    validate_profile(profile)

    qual_path = os.path.join(quality_dir, tag + ".json")
    if os.path.isfile(qual_path):
        with open(qual_path) as f:
            qual_raw = json.load(f)

        qual_latency_tag = qual_raw.get("meta", {}).get("latency_tag")
        if qual_latency_tag and qual_latency_tag != tag:
            print(
                "ERROR: quality sidecar for '{}' has latency_tag='{}'. "
                "This sidecar was generated for a different latency result. "
                "Re-run evaluate/run_eval.py with --latency-result pointing "
                "to the correct file.".format(tag, qual_latency_tag),
                file=sys.stderr,
            )
            sys.exit(1)

        qual_model = qual_raw.get("meta", {}).get("model")
        if qual_model and qual_model != profile["model"]:
            print(
                "ERROR: quality sidecar for '{}' has model='{}' but latency result has model='{}'. "
                "Re-run evaluate/run_eval.py against the correct deployment.".format(
                    tag, qual_model, profile["model"]
                ),
                file=sys.stderr,
            )
            sys.exit(1)

        qm = qual_raw.get("metrics", {})
        profile["quality"] = {
            "overall_score": qm.get("overall_score"),
            "metrics": {k: v for k, v in qm.items() if k != "overall_score"},
        }
        cost = qual_raw.get("cost", {})
        profile["cost"]["per_million_tokens"] = cost.get("per_million_tokens")

        # prefer quality sidecar value; fall back to latency file throughput
        profile["cost"]["throughput_proxy_tokens_per_sec"] = (
            cost.get("throughput_proxy_tokens_per_sec")
            or profile["cost"]["throughput_proxy_tokens_per_sec"]
        )
        profile["_dataset"] = qual_raw.get("meta", {}).get("dataset")
    else:
        print("WARN: no quality sidecar for '{}' (expected {}) — quality metrics will be N/A".format(tag, qual_path), file=sys.stderr)

    return profile


def compute_tradeoff(profiles, baseline_tag):
    """
    Compute relative latency/quality/cost deltas for each profile vs baseline.
    Returns a list of row dicts, one per profile (including baseline).
    """
    baseline = next((p for p in profiles if p["tag"] == baseline_tag), None)
    if baseline is None:
        print(
            "ERROR: baseline tag '{}' not found in profiles. "
            "Available tags: {}".format(baseline_tag, [p["tag"] for p in profiles]),
            file=sys.stderr,
        )
        sys.exit(1)

    datasets = {p["_dataset"] for p in profiles if p.get("_dataset")}
    if len(datasets) > 1:
        print(
            "ERROR: profiles use different eval datasets — quality scores are not comparable. "
            "Datasets found: {}".format(sorted(datasets)),
            file=sys.stderr,
        )
        sys.exit(1)

    base_ttft = baseline["latency"]["ttft_ms_p50"]
    base_quality = (
        baseline["quality"]["overall_score"]
        if baseline.get("quality") and baseline["quality"].get("overall_score") is not None
        else None
    )
    base_cost_pm = baseline["cost"].get("per_million_tokens")
    base_throughput = baseline["cost"].get("throughput_proxy_tokens_per_sec")

    rows = []
    for p in profiles:
        is_baseline = p["tag"] == baseline_tag
        tag_ttft = p["latency"]["ttft_ms_p50"]
        tag_quality = (
            p["quality"]["overall_score"]
            if p.get("quality") and p["quality"].get("overall_score") is not None
            else None
        )
        tag_cost_pm = p["cost"].get("per_million_tokens")
        tag_throughput = p["cost"].get("throughput_proxy_tokens_per_sec")

        if is_baseline:
            latency_imp = None
            quality_delta = None
            cost_red = None
        else:
            latency_imp = (base_ttft - tag_ttft) / base_ttft * 100 if base_ttft else None

            if tag_quality is not None and base_quality is not None:
                quality_delta = (tag_quality - base_quality) * 100
            else:
                quality_delta = None

            if tag_cost_pm is not None and base_cost_pm is not None:
                cost_red = (base_cost_pm - tag_cost_pm) / base_cost_pm * 100
            elif tag_throughput and base_throughput:
                cost_red = (tag_throughput - base_throughput) / tag_throughput * 100
            else:
                cost_red = None

        rows.append({
            "tag": p["tag"],
            "is_baseline": is_baseline,
            "ttft_ms_p50": tag_ttft,
            "throughput_tokens_per_sec": p["latency"]["throughput_tokens_per_sec"],
            "overall_score": tag_quality,
            "cost_per_million": tag_cost_pm,
            "latency_improvement_pct": latency_imp,
            "quality_delta_pct": quality_delta,
            "cost_reduction_pct": cost_red,
        })

    return rows


def recommend(rows, quality_threshold=0.10):
    """
    Filter and rank profiles vs baseline. Returns a Recommendation dict.
    Eliminates profiles with quality_delta_pct < -threshold*100.
    Falls back to baseline if no candidate survives.
    """
    baseline_row = next(r for r in rows if r["is_baseline"])
    baseline_tag = baseline_row["tag"]

    all_quality_none = all(r["quality_delta_pct"] is None for r in rows if not r["is_baseline"])

    warning = None
    if all_quality_none:
        warning = "No quality data available — ranked by latency only"

    augmented = []
    eliminated = []
    for r in rows:
        row = dict(r)
        if r["is_baseline"]:
            row["status"] = "baseline"
        elif (
            r["quality_delta_pct"] is not None
            and r["quality_delta_pct"] < -quality_threshold * 100
        ):
            row["status"] = "eliminated"
            eliminated.append(row)
        else:
            row["status"] = "candidate"
        augmented.append(row)

    candidates = [r for r in augmented if r["status"] == "candidate"]
    candidates.sort(key=lambda r: (r["latency_improvement_pct"] or 0), reverse=True)

    if candidates:
        recommended_tag = candidates[0]["tag"]
        for r in augmented:
            if r["tag"] == recommended_tag:
                r["status"] = "RECOMMENDED"
    else:
        recommended_tag = baseline_tag
        if warning is None:
            warning = "No alternative met the quality threshold — baseline recommended"

    return {
        "recommended_tag": recommended_tag,
        "baseline_tag": baseline_tag,
        "rows": augmented,
        "eliminated": eliminated,
        "warning": warning,
        "quality_threshold": quality_threshold,
    }


def render(recommendation, output_format="markdown"):
    """Render recommendation as a markdown terminal card or raw JSON."""
    if output_format == "json":
        return json.dumps(recommendation, indent=2)

    rec_tag = recommendation["recommended_tag"]
    baseline_tag = recommendation["baseline_tag"]
    rows = recommendation["rows"]
    eliminated = recommendation["eliminated"]
    warning = recommendation.get("warning")
    threshold = recommendation.get("quality_threshold", 0.10)

    rec_row = next((r for r in rows if r["tag"] == rec_tag), None)
    base_row = next((r for r in rows if r["tag"] == baseline_tag), None)

    lines = ["=== Deployment Recommendation ===", ""]
    lines.append("Recommended: {}".format(rec_tag))
    lines.append("")

    if rec_row and rec_row["tag"] != baseline_tag:
        lat = rec_row.get("latency_improvement_pct")
        cost = rec_row.get("cost_reduction_pct")
        qual = rec_row.get("quality_delta_pct")
        base_ttft = base_row["ttft_ms_p50"] if base_row else "N/A"
        rec_ttft = rec_row["ttft_ms_p50"]
        base_cost = "${:.2f}".format(base_row["cost_per_million"]) if base_row and base_row.get("cost_per_million") is not None else "N/A"
        rec_cost = "${:.2f}".format(rec_row["cost_per_million"]) if rec_row.get("cost_per_million") is not None else "N/A"
        base_score = "{:.3f}".format(base_row["overall_score"]) if base_row and base_row.get("overall_score") is not None else "N/A"
        rec_score = "{:.3f}".format(rec_row["overall_score"]) if rec_row.get("overall_score") is not None else "N/A"

        lat_str = "{:.1f}%  ({}ms → {}ms TTFT p50)".format(lat, base_ttft, rec_ttft) if lat is not None else "N/A"
        cost_str = "{:.1f}%  ({} → {} per 1M tokens)".format(cost, base_cost, rec_cost) if cost is not None else "N/A"
        qual_sign = "+" if qual is not None and qual >= 0 else ""
        qual_str = "{}{:.1f}%  ({} → {})".format(qual_sign, qual, base_score, rec_score) if qual is not None else "N/A"

        lines.append("  Latency Improvement:  {}".format(lat_str))
        lines.append("  Cost Reduction:       {}".format(cost_str))
        lines.append("  Quality Delta:        {}".format(qual_str))
        lines.append("")

    if eliminated:
        for e in eliminated:
            qdelta = abs(e["quality_delta_pct"])
            lines.append("Eliminated: {} — quality drop {:.1f}% exceeds threshold ({:.1f}%)".format(
                e["tag"], qdelta, threshold * 100))
        lines.append("")

    if warning:
        lines.append("Warning: {}".format(warning))
        lines.append("")

    lines.append("Tradeoff Table:")
    header = "  {:<20} {:>10}   {:>7}   {:>7}   {:>8}   {}".format(
        "Tag", "TTFT p50", "Tok/s", "Quality", "Cost/1M", "Status")
    lines.append(header)

    for r in rows:
        ttft = "{}ms".format(r["ttft_ms_p50"])
        toks = str(int(r["throughput_tokens_per_sec"])) if r.get("throughput_tokens_per_sec") is not None else "N/A"
        qual = "{:.3f}".format(r["overall_score"]) if r.get("overall_score") is not None else "N/A"
        cost = "${:.2f}".format(r["cost_per_million"]) if r.get("cost_per_million") is not None else "N/A"
        status = r.get("status", "")
        lines.append("  {:<20} {:>10}   {:>7}   {:>7}   {:>8}   {}".format(
            r["tag"], ttft, toks, qual, cost, status))

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Quality-aware deployment advisor — compare latency, quality, and cost."
    )
    parser.add_argument("--tags", nargs="+", required=True,
                        help="Deployment tags to compare")
    parser.add_argument("--baseline", required=True,
                        help="Tag to use as baseline")
    parser.add_argument("--quality-threshold", type=float, default=0.10,
                        help="Max acceptable quality drop vs baseline (default: 0.10 = 10%%)")
    parser.add_argument("--output", choices=["markdown", "json"], default="markdown",
                        help="Output format (default: markdown)")
    parser.add_argument("--latency-dirs", nargs="+",
                        help="Override latency result directories (default: results/synthetic results/real)")
    parser.add_argument("--quality-dir",
                        help="Override quality sidecar directory (default: results/quality)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print what would run, then exit")
    args = parser.parse_args()

    latency_dirs = args.latency_dirs or [
        os.path.join(REPO_ROOT, "results", "synthetic"),
        os.path.join(REPO_ROOT, "results", "real"),
    ]
    quality_dir = args.quality_dir or os.path.join(REPO_ROOT, "results", "quality")

    profiles = []
    for tag in args.tags:
        profile = load_deployment(tag, latency_dirs, quality_dir)
        profiles.append(profile)

    if args.dry_run:
        print("Dry run — would compare {} profiles:".format(len(profiles)))
        for p in profiles:
            qual_status = "quality: {:.3f}".format(p["quality"]["overall_score"]) \
                if p.get("quality") and p["quality"].get("overall_score") is not None \
                else "quality: N/A"
            print("  {} ({}, {})".format(p["tag"], p["model"], qual_status))
        print("Baseline: {}".format(args.baseline))
        print("Quality threshold: {:.0f}%".format(args.quality_threshold * 100))
        sys.exit(0)

    rows = compute_tradeoff(profiles, args.baseline)
    recommendation = recommend(rows, args.quality_threshold)
    print(render(recommendation, args.output))


if __name__ == "__main__":
    main()
