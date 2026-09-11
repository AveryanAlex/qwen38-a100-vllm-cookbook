# CPU KV-cache offloading

The production rollout on 2026-09-11 uses the pinned backport's native CPU cache with a **128 GiB total host-memory budget across TP4**. Validation results are recorded below. The existing `VLLM_PLE_CPU_OFFLOAD=1` concerns the model's PLE table and remains enabled separately.

## Expected behavior and benefit

CPU offloading provides a second cache tier: computed blocks can be saved to RAM, retained after their GPU copies are evicted, and restored to GPU for a matching prefix. This is a best-effort cache; saves can be dropped, and CPU entries can also be evicted when the tier fills. It is not a guarantee that every evicted GPU block is preserved.

The likely benefit is lower time to first token and less repeated prefill for returning conversations or long documents whose prefixes no longer fit in the GPU cache. It may also reduce recomputation after preemption when the required blocks were saved. GPU-resident cache hits already avoid this recomputation, so RAM adds little for those hits. Cold, unique prompts have nothing to reload; transfers and cache bookkeeping may add overhead.

This does not directly increase steady decode tokens/s, and does not make 128 GiB of RAM equivalent to extra VRAM for one active sequence. Restored attention blocks must fit on GPU to execute. It does not by itself enable [1M context](CONTEXT.md) or resolve temporary CUDA allocation pressure.

## Evidence from installed source

Paths below are relative to the installed `vllm/` package in the [pinned image](PROVENANCE.md).

| Source | Finding |
|---|---|
| `config/cache.py`, `config/vllm.py::_post_init_kv_transfer_config` | `kv_offloading_size` is GiB summed across TP ranks; native selects `OffloadingConnector` unless `VLLM_USE_SIMPLE_KV_OFFLOAD` overrides it. |
| `distributed/kv_transfer/kv_connector/v1/offloading_connector.py` | Declares `SupportsHMA`; describes best-effort saves. |
| `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | Handles Mamba state, hybrid groups and Mamba alignment boundaries. |
| `distributed/kv_transfer/kv_connector/v1/offloading/worker.py` | Registers both attention pages and Mamba state pages for transfers. |
| `v1/worker/gpu/kv_connector.py` | V2 runner registers caches and invokes load/store/preemption hooks. |
| `models/qwen4_exp/nvidia/qsa.py` | QSA exposes `FullAttentionSpec`, a type accepted by the native scheduler. |
| `v1/kv_offload/cpu/shared_offload_region.py` | Default host backing is shared memory under `/dev/shm`, with capacity checks and CUDA registration. |

Source compatibility alone is insufficient: re-run the save/restore checks below after changing the image, model, cache layout or patches.

The inspected host had about 503 GiB total RAM, 356 GiB available, and 244 GiB available in `/dev/shm`. The container uses host IPC and has no explicit Podman memory limit. Thus a 128 GiB cache is plausible from the observed host capacity. Availability changes with load; parent cgroup limits, shared-memory space and registration must also be checked at trial time. This is RAM cache, not OS swap or weight offloading.

## Enable and restart

Use the cookbook's patched launcher, including patch 06. The unmodified pinned image's native offloader fails on this model's scratch-cache group; adding the CLI flags alone is insufficient.

1. Check host capacity with `free -h` and `df -h /dev/shm`. Budget the CPU cache in addition to the PLE table, API/worker processes, multimodal caches and other services. The launcher uses host IPC, so `/dev/shm` is shared with the host. A container or parent service memory limit must accommodate the whole deployment, not just the CPU-cache size.
2. In `config.json`, set `"kv_offloading_gib": 128`. Zero disables it. A missing field defaults to zero for compatibility with existing installations. The example config enables 128 GiB.
3. Restart through the existing manager:

```bash
# If installed as the cookbook's systemd service:
systemctl --user restart qwen38-a100.service
python3 scripts/cookbook.py wait --timeout 900

# For manual mode, use this instead:
python3 scripts/cookbook.py start
python3 scripts/cookbook.py wait --timeout 900
```

The launcher supplies these vLLM options:

```text
--enable-prefix-caching
--kv-offloading-size 128
--kv-offloading-backend native
```

Keep `VLLM_USE_SIMPLE_KV_OFFLOAD` unset: the tested connector is `OffloadingConnector`. No LMCache service or extra package is required. The setting persists in `config.json`, so a service restart recreates the same configuration. Restarting the existing container also preserves its arguments. **CPU cache contents are ephemeral and start cold after a model-server restart.** Restarting Podman's API service alone does not restart the model.

## Per-request cache usage in the API

The cookbook launcher includes `--enable-prompt-tokens-details`. This exposes cached-token counts in Chat Completions usage, independently of whether CPU offloading is enabled:

| Endpoint | Cached-token field |
|---|---|
| `/v1/chat/completions` | `usage.prompt_tokens_details.cached_tokens` |
| `/v1/responses` | `usage.input_tokens_details.cached_tokens` |

For streaming Chat Completions, send `"stream_options": {"include_usage": true}` and read the final usage chunk. Responses streaming includes usage in the final response object. The pinned backport already exposes Responses cached-token details without this flag; the flag enables the previously missing Chat Completions details. It does not enable request-content logging.

The cached count includes both local GPU and external/CPU prefix hits. It does not break those tiers out per request. Counts reflect reusable cache boundaries and may be smaller than the identical input prefix. The backport also reports `created_cache_tokens` in Chat Completions details; that field is not a CPU-transfer counter. Use `/metrics` for separate aggregate local/external hits and CPU transfer bytes.

After updating an existing checkout, restart through its usual manager to apply the launcher flag. Verify normal and streaming usage with:

```bash
mkdir -p results
python3 benchmarks/check_cache_usage.py --port 19088 --output results/cache-usage.json
```

This sends a unique prompt three times to each endpoint and requires a numeric cached-token field plus positive cache hits on repeated requests. It does not change the server configuration or clear caches.

Production verification on 2026-09-11 passed all six checks. Chat Completions reported 0 cached tokens initially and 800 on both the normal repeat and streaming repeat of a 2,142-token prompt. Responses reported 800 in both formats, reusing the same prompt already submitted to Chat Completions. Its stream ended with `response.completed`. The flag was the only serving-argument change; 512K context, image/video inputs, CPU-cache capacity and performance settings were preserved. [Recorded usage responses](../measurements/2026-09-11-cache-usage/verification.json) and [deployment audit](../measurements/2026-09-11-cache-usage/audit-public.json).

## Verify actual cache restoration

Check startup logs for `OffloadingConnector` and the CPU shared-memory allocation, then inspect transfer metrics:

```bash
podman logs qwen38-a100 2>&1 | rg 'OffloadingConnector|mmap|offload|ERROR'
curl -fsS http://127.0.0.1:19088/metrics | rg 'kv_offload_(load|store)_bytes_total'
```

Stores prove that blocks reached RAM. Loads must increase during a cache revisit to establish restoration. Counter values are engine-wide and include other clients, so use an otherwise quiet endpoint for precise attribution.

Run the approximately 190K book test against either context configuration (the measurements below used native 262K):

```bash
python3 scripts/prepare_book.py
mkdir -p results
podman exec -e VLLM_PLE_CPU_OFFLOAD=0 qwen38-a100 python3 /cookbook/benchmarks/test_offload_projection.py -v
python3 benchmarks/check_cpu_cache.py --port 19088 --output results/cpu-cache-book.json
```

This issues a unique approximately 190K-token book prompt, six distinct similarly sized churn prompts, and then repeats the original book request. Churn identities differ at the beginning to defeat shared-prefix hits; their total exceeds the tested GPU prefix capacity. It requires increasing CPU-load/external-hit counters, zero GPU prefix hits on the revisit, and all four planted codes in order in both book answers. Review the saved summaries and seven factual answers manually. It uses natural eviction without enabling development/reset APIs. More GPU capacity may require increasing `--churn-requests`.

Also run the existing functional/cache, image/video and throughput checks in README.md. Compare cold and restored TTFT, steady decode and aggregate throughput; prefix restoration is the intended gain. Check server generation counters against benchmark token counts to identify unrelated traffic.

## Disable or recover from a failed startup

Set `"kv_offloading_gib": 0` and restart using the same manager. This removes the offload flags while retaining the model, MTP, graph and multimodal settings. If startup reports insufficient shared memory or host registration/allocation failure, inspect available RAM, `/dev/shm`, and cgroup limits; reduce the cache budget as needed. Do not remove shared-memory objects belonging to running processes.

## Deployment measurements

The patched 128 GiB configuration is running in production. Startup took **393 seconds** after the container started. The shared region was **137.41 GB** (approximately 128 GiB, rounded to cache chunks), interleaved across both NUMA nodes. GPU cache allocation remained **10.55 GiB per rank**. The host had approximately **225 GiB available RAM** after allocation.

| Measurement | Result |
|---|---:|
| Single-request decode, quiet `single_2` scenario | **108.56 tokens/s** |
| Two requests, aggregate | **192.55 tokens/s** |
| Four requests, aggregate | **362.55 tokens/s** |
| Eight requests, aggregate | **610.05 tokens/s** |
| Book prompt | **190,410 tokens** |
| Cold book TTFT | **20.35 s** |
| Book TTFT after GPU eviction, restored from RAM | **1.52 s** |
| CPU-to-GPU bytes during book restore, summed across ranks | **10,907,852,800 bytes (10.16 GiB)** |
| GPU prefix hits during restore | **0 tokens** |
| External/CPU prefix hits during restore | **189,600 tokens** |
| Book decode, cold/restored | **109.49 / 108.60 tokens/s** |

Six distinct **190,291-token** churn prompts forced natural eviction. The revisit's first-token latency was **13.36× lower than the initial cold request in this run**. This is a measured cache-reuse benefit, not a general decode speedup. Those six cold churn requests took 16.75–17.04 s to first token; the initial cold request includes first-use overhead. The restore interval generated exactly the expected 915 tokens server-wide, with no extra client generation observed.

The two book answers were **byte-identical**, ended normally, and passed all four planted-code and seven factual-answer checks. Both summaries still incorrectly attribute presentation of Manette's prison letter to Madame Defarge rather than Ernest Defarge. Four projection unit checks, 13 functional checks, 32 prefix-cache checks, eight image cases and six video/multi-input cases passed their targeted acceptance checks. Video timestamps and incidental details are not perfectly reliable; the saved quality review records limitations.

The first short throughput scenario overlapped other client traffic; use the quiet `single_2` row above. Each displayed parallel scenario's server token delta exactly matched the benchmark's expected output count. The fresh no-offload baseline also overlapped traffic in five of six scenarios, so it is retained as a shared-load record rather than claimed as an isolated A/B control. The earlier quiet multimodal run measured 108.42 single decode and 612.68 aggregate at eight requests, close to this rollout's results.

Twelve allocator OOM warnings occurred during validation, but there was no `OutOfMemoryError`, traceback, engine restart or failed request in the successful rollout. They are consistent with the previously documented recovered allocation pressure; CPU KV offloading does not eliminate temporary GPU allocation pressure. Final health was HTTP 200, `OOMKilled=false`, and the image, arguments, overlay hashes, multimodal settings and native context passed the production audit.

The production launcher and selected experiment profile were updated, so regenerating that launcher retains the cache flags and patch. The existing production installation remains manually managed; this rollout did not replace its old systemd unit. For reproducible automatic startup, install the cookbook service as described in README.md. A container/launcher restart retains the configuration; a successful second full-model restart was not separately benchmarked.

[Raw book restoration evidence](../measurements/2026-09-11-cpu-cache/book-restore.json), [throughput and load attribution](../measurements/2026-09-11-cpu-cache/bench-load-check.json), [quality review](../measurements/2026-09-11-cpu-cache/quality-review.json), and [final audit](../measurements/2026-09-11-cpu-cache/final-audit-public.json).

No 1M-context inference is part of this rollout.

## Startup defect and patch

The first 128 GiB attempt failed before serving with `tokens_per_block=8 not divisible by tokens_per_hash=800`. Its group sizes were `[800, 800, 800, 800, 8, 800]`. The previous launcher was restored automatically. This was a connector configuration failure, not a host/GPU OOM.

The 8-token group is a circular scratch buffer explicitly marked `prefix_cacheable=False`. The GPU cache manager already excludes such scratch rings from reusable prefixes and allocates fresh rings for external cache hits. The native offloader incorrectly included it in the reusable-cache geometry.

Patch 06 projects the offloader's view onto prefix-cacheable groups. It preserves physical block IDs and maps new/resumed request block lists, allocated blocks and Mamba boundary handoffs consistently. The GPU scheduler retains its complete layout. It permits omission only for known circular scratch types and checks that scheduler/hash alignment is unchanged; it does not remove the divisibility assertion. Four CPU-only regression checks cover the mapping, unchanged input data/alignment, identity behavior without scratch groups, and rejection of unknown non-cacheable group types. Actual model save/restore validation remains essential.
