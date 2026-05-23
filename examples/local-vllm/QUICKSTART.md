# Local vLLM Quickstart (5 minutes)

Run `llm-inference-bench` against a vLLM server on your own machine.

## 1. Install

```bash
pip install vllm aiohttp
```

vLLM needs a GPU. For CPU-only environments, skip to the [Docker](#docker-alternative)
or [SGLang](#sglang-alternative) sections, or use synthetic data via
`python data/generate_synthetic.py`.

## 2. Start vLLM

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --port 8000 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --enable-prefix-caching
```

Wait for `Application startup complete` in the log before benchmarking.

## 3. Benchmark

```bash
python collect/run_bench.py \
  --endpoint http://localhost:8000/v1/completions \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --isl 2048 --osl 128 \
  --concurrency 10 --duration 90 \
  --tag local_isl2k_c10
```

Output is written to `results/real/local_isl2k_c10.json`.

## 4. Generate report

```bash
python analyze/report.py --output report.md
```

Open `report.md` in any markdown viewer.

## 5. Get a recommendation

```bash
python playbook/advisor.py \
  --isl 2048 --latency-sla 700 --concurrency 20 \
  --scale mixed --gpu l4 --model-precision fp8
```

---

## Docker alternative

```bash
docker run --gpus all --shm-size=2g -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --max-num-seqs 128
```

Then benchmark exactly the same way against `http://localhost:8000`.

## SGLang alternative

SGLang exposes the same OpenAI-compatible interface:

```bash
python -m sglang.launch_server \
  --model-path mistralai/Mistral-7B-Instruct-v0.3 \
  --port 8000
```

Add `--runtime sglang` to the benchmark command so the metadata reflects it.

## GPU memory by model size (vLLM defaults, FP16)

| Model | Min VRAM | Recommended GPU |
|---|---:|---|
| 7B/8B | 16 GB | L4 (FP8), L40S, A100-40 |
| 13B | 28 GB | A100-40, L40S (FP8) |
| 30B–34B | 70 GB | A100-80, H100 |
| 70B | 140 GB | 2× A100-80, 2× H100, or FP8 on 1× H100 |

L4 is FP8-friendly but tight on VRAM for FP16 8B+. Use FP8 quantized weights on L4.
