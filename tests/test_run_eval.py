import json
import os
import pytest


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


VALID_ROW = {
    "schema_version": 1,
    "id": "test_001",
    "workload": "chat",
    "prompt": "What is 2+2?",
    "expected": "4",
}


class TestLoadDataset:
    def test_valid_rows_are_returned(self, tmp_path):
        p = tmp_path / "test.jsonl"
        write_jsonl(p, [VALID_ROW])
        from evaluate.run_eval import load_dataset
        rows = load_dataset(str(p))
        assert len(rows) == 1
        assert rows[0]["id"] == "test_001"

    def test_missing_required_field_exits(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        bad = {k: v for k, v in VALID_ROW.items() if k != "expected"}
        write_jsonl(p, [bad])
        from evaluate.run_eval import load_dataset
        with pytest.raises(SystemExit):
            load_dataset(str(p))

    def test_invalid_workload_exits(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        bad = {**VALID_ROW, "workload": "unknown_type"}
        write_jsonl(p, [bad])
        from evaluate.run_eval import load_dataset
        with pytest.raises(SystemExit):
            load_dataset(str(p))

    def test_empty_file_exits(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        from evaluate.run_eval import load_dataset
        with pytest.raises(SystemExit):
            load_dataset(str(p))

    def test_invalid_json_exits(self, tmp_path):
        p = tmp_path / "broken.jsonl"
        p.write_text("not json\n")
        from evaluate.run_eval import load_dataset
        with pytest.raises(SystemExit):
            load_dataset(str(p))

    def test_unsupported_schema_version(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text(json.dumps({
            "schema_version": 2, "id": "x", "workload": "chat",
            "prompt": "hi", "expected": "hello"
        }) + "\n")
        from evaluate.run_eval import load_dataset
        with pytest.raises(SystemExit):
            load_dataset(str(p))


class TestNormalizeScore:
    def test_hallucination_is_inverted(self):
        from evaluate.run_eval import normalize_score
        assert normalize_score("hallucination", 0.2) == pytest.approx(0.8)

    def test_hallucination_clamps_at_zero(self):
        from evaluate.run_eval import normalize_score
        assert normalize_score("hallucination", 1.1) == pytest.approx(0.0)

    def test_relevancy_passes_through(self):
        from evaluate.run_eval import normalize_score
        assert normalize_score("answer_relevancy", 0.9) == pytest.approx(0.9)

    def test_correctness_passes_through(self):
        from evaluate.run_eval import normalize_score
        assert normalize_score("correctness", 0.75) == pytest.approx(0.75)


class TestSelectMetrics:
    def test_chat_has_relevancy_and_correctness(self):
        from evaluate.run_eval import select_metrics
        metrics = select_metrics("chat", has_contexts=False)
        assert "answer_relevancy" in metrics
        assert "correctness" in metrics
        assert "faithfulness" not in metrics

    def test_rag_without_contexts_has_no_faithfulness(self):
        from evaluate.run_eval import select_metrics
        metrics = select_metrics("rag", has_contexts=False)
        assert "faithfulness" not in metrics

    def test_rag_with_contexts_adds_faithfulness_and_hallucination(self):
        from evaluate.run_eval import select_metrics
        metrics = select_metrics("rag", has_contexts=True)
        assert "faithfulness" in metrics
        assert "hallucination" in metrics

    def test_long_context_same_as_chat(self):
        from evaluate.run_eval import select_metrics
        assert select_metrics("long_context", False) == select_metrics("chat", False)


class TestDeriveTag:
    def test_strips_json_extension(self):
        from evaluate.run_eval import derive_tag
        assert derive_tag("results/real/vllm_l4fp8_isl2k_c10.json") == "vllm_l4fp8_isl2k_c10"

    def test_works_with_nested_path(self):
        from evaluate.run_eval import derive_tag
        assert derive_tag("/some/deep/path/my_tag.json") == "my_tag"


class TestWriteSidecar:
    def test_writes_valid_json(self, tmp_path):
        from evaluate.run_eval import write_sidecar
        path = write_sidecar(
            out_dir=str(tmp_path),
            tag="test_tag",
            latency_tag="test_tag",
            evaluator="deepeval",
            model="llama-3.1-8b",
            dataset_path="datasets/rag.jsonl",
            num_samples=15,
            metrics={"answer_relevancy": 0.93, "correctness": 0.91},
            overall_score=0.92,
            cost_per_million=0.80,
            throughput_proxy=262,
        )
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert data["meta"]["latency_tag"] == "test_tag"
        assert data["metrics"]["overall_score"] == 0.92
        assert data["cost"]["per_million_tokens"] == 0.80
        assert data["cost"]["throughput_proxy_tokens_per_sec"] == 262

    def test_creates_output_dir_if_missing(self, tmp_path):
        from evaluate.run_eval import write_sidecar
        nested = str(tmp_path / "deep" / "dir")
        write_sidecar(nested, "t", "t", "deepeval", "m", "d", 1, {}, 0.5, None, None)
        assert os.path.isdir(nested)
