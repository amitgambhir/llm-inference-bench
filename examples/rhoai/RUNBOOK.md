# RHOAI Runbook — Llama 3.1 8B FP8 on NVIDIA L4

Step-by-step guide to reproduce the validation run that backs every "real"
data point in this repo. The setup uses Red Hat OpenShift AI (RHOAI) with
KServe RawDeployment on AWS `g6.12xlarge` (4× NVIDIA L4).

> All numbers in BENCHMARK_FINDINGS.md were collected via this runbook.

## 1. Pre-flight

```bash
# Authenticate to your cluster
oc login --token=... --server=https://api.<cluster>.<domain>:6443

# Verify GPU nodes are available
oc describe nodes | grep nvidia.com/gpu

# Pick (or create) a namespace
oc new-project llm-bench

# Optional: generate synthetic reference data so report.py has something to compare to
python data/generate_synthetic.py
```

## 2. Deploy the ServingRuntime

```bash
oc apply -f examples/rhoai/serving_runtime.yaml
oc get servingruntime rhaiis-cuda -n llm-bench
```

The runtime is the recipe for *how* vLLM runs. Apply once per namespace.

## 3. Deploy the InferenceService

```bash
oc apply -f examples/rhoai/isvc.yaml

# Wait for the predictor pod to come up — model pull from the OCI registry
# takes a few minutes on first run.
oc get pods -w | grep llama-vllm-single

# Verify the model is loaded
POD=$(oc get pod -l serving.kserve.io/inferenceservice=llama-vllm-single -o name | head -1)
oc logs $POD --tail=40 | grep "Application startup complete"
```

## 4. Expose an external route

```bash
oc create route edge \
  --service=llama-vllm-single-predictor \
  --port=8000

ENDPOINT=$(oc get route llama-vllm-single-predictor -o jsonpath='{.spec.host}')
curl -s https://${ENDPOINT}/v1/models | python -m json.tool
```

You should see the served model in the response.

## 5. Run a benchmark

```bash
# Auth disabled in the ISVC, so no --token needed.
python collect/run_bench.py \
  --endpoint https://${ENDPOINT}/v1/completions \
  --model llama-vllm-single \
  --isl 2048 --osl 128 \
  --concurrency 10 --duration 90 \
  --tag rhoai_l4fp8_isl2k_c10 \
  --output-dir ~/results
```

If your cluster *does* have auth enabled:

```bash
export TOKEN=$(oc whoami -t)
python collect/run_bench.py ... --token $TOKEN
```

## 6. Patch vLLM parameters between runs

vLLM args live in the ServingRuntime under
`spec.containers[0].args`. The KServe RawDeployment will restart pods on
ServingRuntime changes if you trigger a rollout.

```bash
# Always check the current args first — array indexes shift after each edit
oc get servingruntime rhaiis-cuda -o jsonpath='{.spec.containers[0].args}' | python -m json.tool
```

### Add a flag (append to the array)

```bash
oc patch servingruntime rhaiis-cuda --type=json \
  -p='[{"op":"add","path":"/spec/containers/0/args/-","value":"--enable-chunked-prefill"}]'
```

### Change a value (replace at a known index)

```bash
# After confirming with oc get that --max-num-seqs lives at index 4:
oc patch servingruntime rhaiis-cuda --type=json \
  -p='[{"op":"replace","path":"/spec/containers/0/args/4","value":"--max-num-seqs=128"}]'
```

### Apply the change

```bash
oc rollout restart deployment/llama-vllm-single-predictor
oc rollout status deployment/llama-vllm-single-predictor
```

Wait for the new pod to reach `Application startup complete`, then re-run
the benchmark with a new `--tag` so the JSON files don't collide.

## 7. Teardown and report

```bash
# Generate the report locally — no GPU needed
python analyze/report.py --output report.md

# Optionally pull JSON back to your workstation
scp -r bastion:~/results/* ./results/real/

# Tear down when finished
oc delete inferenceservice llama-vllm-single
oc delete servingruntime rhaiis-cuda
oc delete route llama-vllm-single-predictor
```

---

## Troubleshooting

**Pod Pending — no GPU available**

```bash
oc describe nodes | grep -A2 nvidia.com/gpu
oc describe pod $POD | tail -20    # look for taints/affinity issues
```

If GPUs are accounted for but no scheduling fit, check if an old `LLMInferenceService`
or `InferenceModel` CRD is holding them (see next).

**Deployments respawning after `oc delete`**

The MaaS / LLM-d operators install extra CRDs that recreate deployments. If
deleting your ISVC isn't enough:

```bash
oc get inferencemodel,llminferenceservice -A
oc delete inferencemodel <name>
oc delete llminferenceservice <name>
```

These CRDs keep recreating the underlying `Deployment`. Removing them stops the loop.

**Connection refused via port-forward**

vLLM listens on container port `8000`. Use:

```bash
oc port-forward svc/llama-vllm-single-predictor 8000:8000
```

Not `8000:80`. The latter is a common copy-paste from HTTP examples and silently
fails with "connection refused".

**401 Unauthorized**

Token expired. Refresh:

```bash
export TOKEN=$(oc whoami -t)
```

If auth is supposed to be disabled, check the ISVC annotation:

```bash
oc get isvc llama-vllm-single -o jsonpath='{.metadata.annotations.security\.opendatahub\.io/enable-auth}'
# should be "false"
```

**Model pull is slow / OOM during pull**

The OCI modelcar image is multi-GB. Ensure the node has enough ephemeral storage,
and that `imagePullPolicy: IfNotPresent` is honored (it's the default).
