# Context limit: 262K, not 1M

The current production deployment and cookbook use **262,144 total tokens** per request. This budget includes the prompt, visual tokens, reasoning and generated output. Image/video item counts of 999 do not change this token limit.

The checkpoint config has max_position_embeddings=262144, rope_type=default and rope_theta=10000000. No YaRN or other context-extension scaling is configured. The live /v1/models response reports max_model_len=262144. The model card also describes its validation at the native 262,144-token shared context, without YaRN.

## API tests on 2026-09-11

| Test | Locally counted prompt tokens | Requested output | Total budget | Result |
|---|---:|---:|---:|---|
| One token beyond native context | 262,144 | 1 | 262,145 | HTTP 400 |
| Full 1M context budget | 1,048,512 | 64 | 1,048,576 | HTTP 400 |
| Small arithmetic control | 26 | 32 | 58 | HTTP 200, correct answer 323 |

The tokenizer from the checkpoint counted the actual rendered chat template. The API's bounded validator reported an **at least** input count once it had enough tokens to reject the request; that lower-bound message does not contradict the independent full count. No 1M-token model prefill, decode or answer-quality test ran, because admission validation rejected the request. Production health remained HTTP 200 and no serving configuration was changed.

The existing successful 190,422-token book experiment is a separate inference/quality test; it does not establish 1M support.

## Reproduce the admission test

This script uses Transformers inside the serving container and assumes the model is mounted at /model. It sends a synthetic repeated-token prompt to verify rejection; it is not a long-context retrieval benchmark. Copy the script into a running deployment and run it, adjusting the container name or port if needed:

```bash
podman cp benchmarks/check_context_limit.py qwen38-a100:/tmp/check_context_limit.py
podman exec -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e TOKENIZERS_PARALLELISM=false \
  qwen38-a100 python3 /tmp/check_context_limit.py --port 19088 --output /tmp/context-limit-check.json
podman cp qwen38-a100:/tmp/context-limit-check.json results/context-limit-check.json
```

Create results/ first. The script deliberately expects the current 262K boundary. If you experiment with another context configuration, change the test expectations before using it.

## What 1M would require

Changing max-model-len alone would not establish working 1M support. An extension experiment would need a model-appropriate positional-scaling configuration, enough KV/state and temporary-buffer memory, and actual full-context retrieval/answer-quality tests. The current startup reports about 769,519 aggregate KV-cache token capacity with BF16 KV, below a million-token request. Existing memory headroom is already tight, as documented in VISION.md. Cache precision/offload, graph memory or other provisioning changes would need validation together with the extension.

The cookbook therefore continues to advertise the tested native limit. It does not claim this checkpoint can never be extended; it records that **1M is neither configured nor validated here**.

[Raw admission-test results](../measurements/2026-09-11-context/context-limit-check.json).

## Extension feasibility checks

The pinned backport's get_rope supports rope_type=yarn together with mrope_section and mrope_interleaved, preserving Qwen's multimodal position layout. A direct CPU-reference rotary-operation test used factor=4, original_max_position_embeddings=262144, and positions up to **1,048,575**. Both one-dimensional text positions and three-axis multimodal positions produced finite outputs. This tests the rotary implementation only, not model inference, retrieval or answer quality.

The resulting rotary cache has shape [4194304,64] in BF16 (512 MiB): this MRoPE implementation reserves extra positional range for multimodal input. That overhead also needs to be included in a memory plan.

The current NVIDIA QSA backend explicitly permits only auto/BF16 main KV and raises NotImplementedError for FP8 storage or KV quantization. The model card's calibrated FP8 scales do not remove this restriction in the active implementation. Enabling FP8 KV would require backend/kernel work or another compatible runtime, followed by numerical and model-quality validation.

A rough linear estimate from the reported cache capacity is 10.55 GiB × 1,048,576 / 769,519 ≈ **14.4 GiB per GPU** for the KV allocation, before other memory requirements. This is an estimate, not a validated 1M allocation plan. A BF16 trial could try freeing graph/draft-model memory, reducing concurrent sequences, and reducing prefill chunk size to lower temporary-buffer peaks. Performance would need to be measured again, and fitting memory would still not prove useful 1M answer quality.

The implementation therefore has a plausible 4× extension path, but full 1M inference remains untested and is not enabled in production. [Rotary-operation evidence](../measurements/2026-09-11-context/yarn-position-check.json).
