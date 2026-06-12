import json
import pytest
from analyze.deployment_advisor import load_deployment, compute_tradeoff


def make_latency_json(tag, ttft_p50=115, ttft_p95=133, throughput=262, model="llama-3.1-8b"):
    return {
        "meta": {
            "tag": tag, "model": model, "runtime": "vllm",
            "gpu": {"name": "NVIDIA L4", "memory_mb": 23034, "util_pct": 0},
            "config": {"chunked_prefill": False, "tensor_parallel_size": 1,
                       "shared_prefix": False},
            "workload": {"isl_approx": 2048, "osl_max": 128, "concurrency": 10,
                         "duration_secs": 90},
            "synthetic": True, "timestamp": "2026-06-12T00:00:00+00:00",
        },
        "metrics": {
            "ttft_ms": {"p50": ttft_p50, "p95": ttft_p95, "p99": 200, "mean": 110},
            "total_latency_ms": {"p50": 4000, "p95": 4500, "p99": 5000},
            "throughput_tokens_per_sec": throughput,
            "throughput_req_per_sec": 2.0,
            "total_requests": 100, "successful_requests": 100, "failed_requests": 0,
        },
    }


def make_quality_json(tag, overall_score=0.93, cost=0.80, throughput=262,
                      model="llama-3.1-8b", latency_tag=None, dataset="datasets/rag.jsonl"):
    return {
        "meta": {
            "tag": tag,
            "latency_tag": latency_tag or tag,
            "evaluator": "deepeval",
            "model": model,
            "dataset": dataset,
            "num_samples": 15,
            "timestamp": "2026-06-12T00:00:00+00:00",
        },
        "metrics": {
            "answer_relevancy": 0.93,
            "correctness": 0.92,
            "overall_score": overall_score,
        },
        "cost": {
            "per_million_tokens": cost,
            "throughput_proxy_tokens_per_sec": throughput,
        },
    }


def write_json(directory, filename, data):
    path = directory / filename
    path.write_text(json.dumps(data))
    return str(path)


def make_profile(tag, ttft_p50=200, throughput=200, overall_score=0.90,
                 cost_per_million=1.20, throughput_proxy=200, is_baseline=False):
    return {
        "tag": tag,
        "model": "llama-3.1-8b",
        "latency": {
            "ttft_ms_p50": ttft_p50,
            "ttft_ms_p95": ttft_p50 + 20,
            "throughput_tokens_per_sec": throughput,
        },
        "quality": {"overall_score": overall_score, "metrics": {}} if overall_score is not None else None,
        "cost": {
            "per_million_tokens": cost_per_million,
            "throughput_proxy_tokens_per_sec": throughput_proxy,
        },
        "_dataset": "datasets/rag.jsonl",
    }


@pytest.fixture
def lat_dir(tmp_path):
    d = tmp_path / "real"
    d.mkdir()
    return d


@pytest.fixture
def qual_dir(tmp_path):
    d = tmp_path / "quality"
    d.mkdir()
    return d


class TestLoadDeployment:
    def test_latency_only_profile(self, lat_dir, qual_dir):
        write_json(lat_dir, "fp16.json", make_latency_json("fp16"))
        profile = load_deployment("fp16", [str(lat_dir)], str(qual_dir))
        assert profile["tag"] == "fp16"
        assert profile["latency"]["ttft_ms_p50"] == 115
        assert profile["latency"]["ttft_ms_p95"] == 133
        assert profile["latency"]["throughput_tokens_per_sec"] == 262
        assert profile["quality"] is None

    def test_flattens_nested_latency_schema(self, lat_dir, qual_dir):
        write_json(lat_dir, "fp16.json", make_latency_json("fp16", ttft_p50=200, ttft_p95=350))
        profile = load_deployment("fp16", [str(lat_dir)], str(qual_dir))
        assert profile["latency"]["ttft_ms_p50"] == 200
        assert profile["latency"]["ttft_ms_p95"] == 350

    def test_merges_quality_sidecar(self, lat_dir, qual_dir):
        write_json(lat_dir, "fp8.json", make_latency_json("fp8", ttft_p50=80))
        write_json(qual_dir, "fp8.json", make_quality_json("fp8", overall_score=0.93))
        profile = load_deployment("fp8", [str(lat_dir)], str(qual_dir))
        assert profile["quality"]["overall_score"] == 0.93
        assert profile["cost"]["per_million_tokens"] == 0.80

    def test_missing_latency_tag_exits(self, lat_dir, qual_dir):
        with pytest.raises(SystemExit):
            load_deployment("nonexistent", [str(lat_dir)], str(qual_dir))

    def test_latency_tag_mismatch_is_hard_error(self, lat_dir, qual_dir):
        write_json(lat_dir, "fp8.json", make_latency_json("fp8"))
        stale = make_quality_json("fp8", latency_tag="different_tag")
        write_json(qual_dir, "fp8.json", stale)
        with pytest.raises(SystemExit):
            load_deployment("fp8", [str(lat_dir)], str(qual_dir))

    def test_real_overrides_synthetic(self, tmp_path):
        syn_dir = tmp_path / "synthetic"
        real_dir = tmp_path / "real"
        qual_dir = tmp_path / "quality"
        syn_dir.mkdir()
        real_dir.mkdir()
        qual_dir.mkdir()
        write_json(syn_dir, "fp8.json", make_latency_json("fp8", ttft_p50=150))
        write_json(real_dir, "fp8.json", make_latency_json("fp8", ttft_p50=90))
        profile = load_deployment("fp8", [str(syn_dir), str(real_dir)], str(qual_dir))
        assert profile["latency"]["ttft_ms_p50"] == 90

    def test_missing_required_latency_field_exits(self, lat_dir, qual_dir):
        data = make_latency_json("fp16")
        del data["metrics"]["ttft_ms"]
        write_json(lat_dir, "fp16.json", data)
        with pytest.raises(SystemExit):
            load_deployment("fp16", [str(lat_dir)], str(qual_dir))

    def test_quality_sidecar_null_overall_score(self, lat_dir, qual_dir):
        write_json(lat_dir, "fp16.json", make_latency_json("fp16"))
        sidecar = make_quality_json("fp16", overall_score=None)
        write_json(qual_dir, "fp16.json", sidecar)
        profile = load_deployment("fp16", [str(lat_dir)], str(qual_dir))
        assert profile["quality"]["overall_score"] is None


class TestComputeTradeoff:
    def test_baseline_row_has_none_deltas(self):
        profiles = [make_profile("base")]
        rows = compute_tradeoff(profiles, "base")
        base_row = rows[0]
        assert base_row["is_baseline"] is True
        assert base_row["latency_improvement_pct"] is None
        assert base_row["quality_delta_pct"] is None
        assert base_row["cost_reduction_pct"] is None

    def test_latency_improvement_computed_correctly(self):
        profiles = [make_profile("base", ttft_p50=200), make_profile("fast", ttft_p50=100)]
        rows = compute_tradeoff(profiles, "base")
        fast_row = next(r for r in rows if r["tag"] == "fast")
        assert fast_row["latency_improvement_pct"] == pytest.approx(50.0)

    def test_quality_delta_computed_correctly(self):
        profiles = [make_profile("base", overall_score=0.90), make_profile("alt", overall_score=0.85)]
        rows = compute_tradeoff(profiles, "base")
        alt_row = next(r for r in rows if r["tag"] == "alt")
        assert alt_row["quality_delta_pct"] == pytest.approx(-5.0)

    def test_cost_reduction_uses_per_million_tokens(self):
        profiles = [
            make_profile("base", cost_per_million=1.20),
            make_profile("cheap", cost_per_million=0.90),
        ]
        rows = compute_tradeoff(profiles, "base")
        cheap_row = next(r for r in rows if r["tag"] == "cheap")
        assert cheap_row["cost_reduction_pct"] == pytest.approx(25.0)

    def test_cost_reduction_falls_back_to_throughput_proxy(self):
        profiles = [
            make_profile("base", cost_per_million=None, throughput_proxy=200),
            make_profile("fast", cost_per_million=None, throughput_proxy=400),
        ]
        rows = compute_tradeoff(profiles, "base")
        fast_row = next(r for r in rows if r["tag"] == "fast")
        assert fast_row["cost_reduction_pct"] == pytest.approx(50.0)

    def test_cost_none_when_no_cost_data(self):
        profiles = [
            make_profile("base", cost_per_million=None, throughput_proxy=None),
            make_profile("alt", cost_per_million=None, throughput_proxy=None),
        ]
        rows = compute_tradeoff(profiles, "base")
        alt_row = next(r for r in rows if r["tag"] == "alt")
        assert alt_row["cost_reduction_pct"] is None

    def test_quality_delta_none_when_no_quality_data(self):
        profiles = [
            make_profile("base", overall_score=None),
            make_profile("alt", overall_score=None),
        ]
        rows = compute_tradeoff(profiles, "base")
        alt_row = next(r for r in rows if r["tag"] == "alt")
        assert alt_row["quality_delta_pct"] is None

    def test_missing_baseline_tag_exits(self):
        profiles = [make_profile("fp8")]
        with pytest.raises(SystemExit):
            compute_tradeoff(profiles, "nonexistent")
