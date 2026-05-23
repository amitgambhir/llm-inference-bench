# Baseten Quickstart

Benchmark a model deployed on [Baseten](https://www.baseten.co/) using
its OpenAI-compatible endpoint.

## Prerequisites

1. A deployed model on Baseten. Most LLMs in the Baseten Model Library
   (Llama, Mistral, Qwen, etc.) expose `/v1/completions` and `/v1/chat/completions`.
2. Your Baseten API key from [app.baseten.co/settings/account/api_keys](https://app.baseten.co/settings/account/api_keys).
3. Your model ID from the model's dashboard URL (`app.baseten.co/models/{MODEL_ID}`).

## Run the benchmark

```bash
export BASETEN_API_KEY=...   # from app.baseten.co
MODEL_ID=abc123              # from the dashboard URL
MODEL_NAME=llama-3-8b-instruct   # the served-model-name (check the model README)

python collect/run_bench.py \
  --endpoint https://model-${MODEL_ID}.api.baseten.co/production/v1/completions \
  --model ${MODEL_NAME} \
  --token ${BASETEN_API_KEY} \
  --isl 2048 --osl 128 \
  --concurrency 10 --duration 90 \
  --tag baseten_isl2k_c10
```

The `--token` flag attaches `Authorization: Bearer ...` to every request.

## Where to find the endpoint URL

In your Baseten model dashboard, the "API" tab shows the curl example. The URL
shape is:

```
https://model-{MODEL_ID}.api.baseten.co/production/v1/completions
```

For development deployments, replace `production` with `development`.

## Latency-optimized vs throughput-optimized deployments

Baseten exposes both deployment modes. To compare them with this tool:

```bash
# 1. Deploy two versions of the same model — one latency-optimized, one throughput-optimized.
# 2. Benchmark each:

python collect/run_bench.py \
  --endpoint https://model-${LAT_ID}.api.baseten.co/production/v1/completions \
  --model ${MODEL_NAME} --token ${BASETEN_API_KEY} \
  --isl 2048 --concurrency 10 --duration 90 \
  --tag baseten_latency_optimized

python collect/run_bench.py \
  --endpoint https://model-${TPUT_ID}.api.baseten.co/production/v1/completions \
  --model ${MODEL_NAME} --token ${BASETEN_API_KEY} \
  --isl 2048 --concurrency 50 --duration 90 \
  --tag baseten_throughput_optimized

# 3. Compare:
python analyze/report.py
```

Latency-optimized typically pins lower `max-num-seqs` and prefers fast first tokens.
Throughput-optimized prefers higher batch sizes and continuous batching. The
right choice depends on whether your workload values TTFT or aggregate tok/s.

## Cold-start note

Baseten autoscales replicas down to zero when idle. If your benchmark starts
against a cold model, the first few requests will include cold-start latency.
Warm the deployment with a handful of curl requests before launching the
benchmark, or set min replicas > 0 in the deployment settings.
