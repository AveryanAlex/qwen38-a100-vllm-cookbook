# 512K context extension

The new example configuration selects **524,288 total tokens** using **2× YaRN**. The checkpoint remains natively 262,144 tokens; no checkpoint files are edited. The prompt, image/video tokens, reasoning and generated output share this total budget.

The extension retains TP4, expert parallelism, MTP1, CUDA graphs, eight maximum sequences, 4096-token prefill batches, BF16 KV and the 128 GiB CPU cache. Increasing the admission limit does not reserve a separate 512K cache for every request: active requests share the GPU cache. Long requests can affect other requests' latency and cache capacity.

## Enable or return to native context

In `config.json`, set:

```json
"max_model_len": 524288
```

Use `262144` to return to native positions without YaRN. Existing configs missing this field default to native 262K; the new example config uses 512K. Other sizes are deliberately rejected by the cookbook until validated.

Restart through your existing manager:

```bash
# Cookbook systemd installation:
systemctl --user restart qwen38-a100.service
python3 scripts/cookbook.py wait --timeout 1000

# Manual installation: use this instead of the systemd restart.
python3 scripts/cookbook.py start
python3 scripts/cookbook.py wait --timeout 1000
```

Changing context requires model-server recreation and starts the GPU/CPU prefix caches cold. Keep the model and state paths on persistent disk. The production runtime launcher and the portable cookbook service are separate installations; the cookbook's systemd setup is in README.md.

For 512K, the launcher adds `--max-model-len 524288` and the following `--hf-overrides` JSON:

```json
{
  "text_config": {
    "rope_parameters": {
      "rope_type": "yarn",
      "rope_theta": 10000000,
      "factor": 2.0,
      "original_max_position_embeddings": 262144,
      "partial_rotary_factor": 0.25,
      "mrope_section": [11, 11, 10],
      "mrope_interleaved": true
    }
  }
}
```

The override merges into the text config, preserving the other architecture fields and the native `max_position_embeddings=262144`. The MRoPE fields preserve the multimodal positional layout. YaRN changes the model's positional behavior globally, including short requests; it is not a per-request switch. Native and extended configurations need separate quality measurements.

## Reproduce full-context validation

Check the advertised limit and health:

```bash
curl -fsS http://127.0.0.1:19088/v1/models
curl -fsS http://127.0.0.1:19088/health
```

The following runs inside the pinned image so it uses the actual checkpoint tokenizer. The repository contains the exact compressed public-domain source texts, with URLs and SHA256 hashes in `benchmarks/fixtures/extended-context-sources.json`.

```bash
podman exec qwen38-a100 python3 /cookbook/benchmarks/check_extended_context.py \
  --port 19088 --output /state/extended-context.json
podman cp qwen38-a100:/state/extended-context.json results/extended-context.json
```

Create `results/` first. The script constructs **522,240 prompt tokens plus 2,048 allowed output tokens = 524,288** using the rendered chat template. It includes the complete *A Tale of Two Cities*, an earlier *Moby Dick* excerpt and a later *Pride and Prejudice* excerpt. It places eight ledger codes across the corpus, asks for their order, a summary of the complete novel and seven factual answers. The complete novel straddles the old native boundary.

It first checks that requesting one extra output token is rejected. It then runs cold and cached full-context requests, requiring the server's prompt-token count to match the independent count, normal completion and all eight codes in order. Inspect the saved summaries and factual answers manually. The model may stop before using all 2,048 allowed output tokens; this tests the full context budget and a prompt close to the boundary, not forced generation of every remaining position.

Use `--prepare-only` to build/count the corpus without sending inference requests. This caught a test-harness issue during development: this Transformers version can return a mapping from `apply_chat_template(tokenize=True)`, whose length is not a token count. The final script explicitly renders text and tokenizes it.

Also run README.md's functional, prefix-cache, image/video and single/parallel throughput checks. Use an otherwise quiet endpoint for a controlled performance comparison; recorded server token counters help detect unrelated generation.

## Results

Production is running **524,288 total tokens with 2× YaRN**, with the previous performance settings and CPU cache retained. The selected runtime profile is `18-yarn-512k`. The launcher was regenerated from that profile and its arguments, image, overlay hashes, context and multimodal settings matched the live audit.

| Check | Observed result |
|---|---|
| Exact prompt + allowed output budget | **522,240 + 2,048 = 524,288 tokens** |
| One token over the total budget | HTTP 400 |
| Full-length generation | HTTP 200, **912 output tokens**, normal stop |
| Codes across the context | **8/8 retrieved**, including the code around token 511,808 |
| Strict code ordering | **Failed:** the sixth and seventh codes were swapped |
| Complete-book summary | Coherent main plot, sacrifice and ending; no mixing of the surrounding novels |
| Seven targeted plot questions | **7/7 correct** |
| Functional / prefix-cache checks | **13/13 / 32/32 passed** |
| Image / video and multi-input cases | **8 / 6 targeted checks passed** |
| Final API health | HTTP 200 |

**Successful long-context execution and retrieval do not imply perfect answer quality.** The order error is preserved as a failed strict check; it was not changed into a pass. A follow-up cold/cached comparison was started, then its client was cancelled when the user requested skipping quiet-window work. There is no completed cached 512K answer or quiet full-context measurement from this rollout. The model server remained running.

The completed long request overlapped other production traffic: server counters recorded **1,844 generated tokens versus this request's 912**, and **six preemptions** during its interval. Its observed TTFT was **127.52 s**, elapsed time **135.93 s**, and decode throughput **108.83 tokens/s**. Treat these as shared-load observations, not isolated latency guarantees or proof that preemption caused the ordering error.

### Normal-request performance

All rows use the same 256-output-token workload. Baseline rows had no additional generation observed. The first three extended-context scenarios overlapped other clients; the remaining displayed scenarios' server generation counts matched the benchmark exactly.

| Scenario | Native 262K | Extended 512K | Load during extended measurement |
|---|---:|---:|---|
| Single decode, repeat 1 | 108.39 tps | 101.76 tps | Shared |
| Single decode, repeat 2 | 108.33 tps | 93.93 tps | Shared |
| Two requests, aggregate | 194.29 tps | 189.13 tps | Shared |
| Medium-prompt decode | 109.27 tps | 107.87 tps | No extra generation observed |
| Four requests, aggregate | 363.79 tps | 369.38 tps | No extra generation observed |
| Eight requests, aggregate | 612.34 tps | 636.54 tps | No extra generation observed |

These observations show normal serving remains functional with the optimization settings intact. The shared singleton measurements do not establish either an isolated slowdown or unchanged single-request performance. No quiet-window waiting or further throughput benchmarking was performed after the user's instruction to stop that work.

### Memory and deployment notes

Startup reported **10.3 GiB GPU cache per rank**, **765,664 aggregate cache tokens** and **1.46× nominal concurrency at 524,288 tokens**. Eight maximum sequences therefore does not mean eight full-length requests fit at once. The native baseline reported 10.55 GiB and 769,519 tokens; model startup profiling and extension overhead changed the resulting pool without changing the configured GPU-memory utilization.

The initial shutdown exceeded the 60-second grace period and Podman reported a stop timeout. Once the old container was confirmed exited, deployment resumed successfully; model startup then took **423 seconds**. The successful extended server logged 12 recovered allocator warnings but no `OutOfMemoryError`, traceback or engine error in the captured audit; `OOMKilled=false` and restart count was zero. CPU offloading does not eliminate temporary GPU allocation pressure.

The rollout did not install a replacement production systemd unit. Its existing runtime launcher and selected profile persist the configuration. New cookbook installations should use README.md's systemd setup for automatic startup. Cache contents start cold after restart.

[Completed full-context evidence](../measurements/2026-09-11-512k/full-context-shared-load.json), [saved answer](../measurements/2026-09-11-512k/full-context.answer.md), [quality review](../measurements/2026-09-11-512k/quality-review.json), [throughput/load attribution](../measurements/2026-09-11-512k/load-attribution.json), and [final audit](../measurements/2026-09-11-512k/final-audit-public.json).

## Limits and earlier experiments

The [native-context and 1M feasibility history](CONTEXT_HISTORY.md) records the earlier 262K admission tests, the 190K book runs, and rotary-only 4× YaRN checks. Those checks did not run a 1M model request. This recipe does not enable 1M, and CPU prefix offloading does not provide active GPU-cache overflow into RAM.
