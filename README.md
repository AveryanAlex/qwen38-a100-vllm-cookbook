# Qwen3.8 A100 vLLM Cookbook

Run **`leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32` on four A100 GPUs** with a pinned vLLM backport, the required SM80 patches, CUDA graphs, and MTP speculative decoding.

On the tested **4 × A100-SXM4-40GB / NVLink** machine, matched single-request throughput increased from **4.78 to 107.88 output tokens/s** (22.6×). Eight simultaneous requests delivered approximately **615 tokens/s in total**. A complete novel with a **190,422-token prompt** took **16.78 seconds to first token**, then decoded at **108.87 tokens/s**.

**Image and video are now enabled by default in the example configuration**, with **999 items per modality per request**. The verified multimodal deployment retained **108.42 single-request tokens/s** and **612.68 aggregate tokens/s** at eight requests. [Image/video setup, tests and limitations](docs/VISION.md).

**The example configuration also enables a 128 GiB native CPU KV cache.** The verified 190K book revisit after GPU eviction reached its first token in **1.52 s**, versus **20.35 s cold**, while single decode remained **108.56 tokens/s**. The recipe includes the required scratch-cache compatibility patch. [Setup, results and limitations](docs/CPU_CACHE.md).

**512K context is enabled in the example configuration with 2× YaRN.** Production completed a 522,240-token prompt, retrieved all eight reference codes and answered seven book questions correctly. It swapped two codes in order; that quality limitation and the shared-load measurements are preserved in the [context runbook](docs/CONTEXT.md).

This is an independently maintained cookbook for this specific checkpoint and runtime. The image's `v0.13.0-sm80` tag is a **backport version**, not upstream vLLM 0.13. Do not substitute a current upstream wheel or `latest` container tag and expect the same behavior.

## What you get

- A step-by-step setup using Podman and NVIDIA CDI.
- Seven reviewed source overlays and the exact patches that produce them.
- A portable launcher and a systemd user service for boot startup and recovery.
- A source build for the optional tuned all-reduce extension.
- Image and video input, 999-item caps, corrected preprocessing, and visual/temporal grounding checks.
- Benchmarks for single/parallel requests, cache reuse, tool calling, thinking, cancellation, and a near-200K book summary.
- [Experiment results](docs/RESULTS.md), [patch explanations](docs/PATCHES.md), [kernel study](docs/CUSTOM_ALL_REDUCE.md), and [recorded measurement data](measurements/2026-09-11).

## 1. Check the hardware and software

The measured configuration was:

| Component | Tested configuration |
|---|---|
| GPUs | 4 × NVIDIA A100-SXM4-40GB, SM80 |
| GPU links | All pairs connected by NV4 links |
| CPU | 2 × AMD EPYC 7H12 |
| Host RAM | Approximately 503 GiB installed |
| OS | Ubuntu 22.04, Linux 6.8 |
| NVIDIA driver | 580.126.09 |
| Runtime | Rootless Podman 5.8.4 with NVIDIA CDI |
| Model | AWQ W4A16 g32 routed experts; official FP8 PLE table offloaded to pinned CPU memory |
| KV cache | BF16; 524,288 total context with 2× YaRN (native: 262,144) |

Have at least **200 GiB of free fast disk** for the approximately 129 GiB checkpoint, image, and compiler caches. Initialization and the large CPU-resident PLE table require substantial host memory; the historical setup guidance was roughly 100 GiB free at minimum, but that minimum was not validated as a capacity target. Use ample headroom; the measurements came from a 503 GiB host. Four PCIe A100s without the same links may perform very differently. Smaller GPU configurations and eight concurrent 200K prompts were not tested.

Install Python **3.10+**, Git, curl, Podman, and a working NVIDIA driver/container toolkit. Python benchmark clients use only the standard library; Torch, Transformers, CUDA and the compiler run inside the pinned image. On Ubuntu, base tools can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y python3 git curl podman uidmap slirp4netns fuse-overlayfs
```

Follow [NVIDIA's container-toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for the driver/toolkit setup. If the toolkit has not already generated CDI devices:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list
nvidia-smi
nvidia-smi topo -m
podman run --rm --device nvidia.com/gpu=all docker.io/library/ubuntu:22.04 nvidia-smi
```

Ubuntu 22.04’s original Podman package may be too old for CDI; install a newer supported Podman release if that command fails. Use a Podman release with `--device nvidia.com/gpu=all` support. Continue only when the container sees all four GPUs. Consult the [CDI guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html) if device resolution fails. On SELinux hosts, bind-mount labeling needs host-specific configuration; this recipe was verified on Ubuntu.

## 2. Get the cookbook and configure paths

Download or clone this repository, then enter its directory:

```bash
cd qwen38-a100-vllm-cookbook
cp config.example.json config.json
```

Edit `config.json` for your machine:

```json
{
  "model_dir": "~/models/leoncca-awq",
  "state_dir": "~/.local/share/qwen38-a100",
  "container_name": "qwen38-a100",
  "port": 19088,
  "tuned_all_reduce": true,
  "vision": true,
  "kv_offloading_gib": 128,
  "max_model_len": 524288
}
```

The launcher expands `~/`. Use absolute paths otherwise. Keep this checkout and the state directory on persistent disk: the service mounts source overlays directly from the checkout. Generated model files, caches, local configuration, binaries, and new results are ignored by Git.

`vision: true` enables **image and video input** with the processor settings described in the [multimodal guide](docs/VISION.md). Both item counts are set to **999 per request**; context and memory still bound real requests. Video uses explicit frame and pixel budgets, described in the guide. Set it to `false` for the original text-only recipe. Existing configuration files without `vision` preserve text-only behavior.

`kv_offloading_gib: 128` reserves approximately **128 GiB total host RAM across TP4** for the native CPU KV cache. It retains reusable prefixes beyond GPU-cache eviction. Budget this in addition to the model's CPU PLE table and other host memory, and ensure `/dev/shm` has sufficient space. Set it to `0` to disable; existing configs without the field default to `0`. See the [CPU cache runbook](docs/CPU_CACHE.md) for setup, verification and performance limits.

`max_model_len: 524288` enables **512K total context with 2× YaRN**. The launcher preserves the checkpoint's native 262K position setting and supplies the multimodal-aware scaling override. Set this field to `262144` for native positions without scaling; existing configs without it remain native. The budget includes prompt, visual, reasoning and output tokens. See the [context-extension runbook](docs/CONTEXT.md).

`tuned_all_reduce: true` selects the measured reduction-kernel tuning. Set it to `false` to use the existing custom all-reduce kernel and skip the extension build; this still includes the correctness fix and main speedups, reaching approximately 107.8 decode tokens/s in the matched 256-token tests. The extension adds about **0.62%**, not the main 22× gain.

Pull the exact image:

```bash
python3 scripts/cookbook.py pull
python3 scripts/cookbook.py check-image
```

The launcher pins:

```text
docker.io/lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3
```

`check-image` checks original source hashes inside the image and syntax-checks all shipped overlays. Nothing is installed into the image. [manifest.json](manifest.json) records the image, source commit, model revision, and upstream/image/patched file identities.

## 3. Download and verify the complete checkpoint

Read the model's [Qwen Community license](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32/blob/fa56146238f9fcd5ab591b7052b31e28efdca5c3/LICENSE). Model weights have their own license; this cookbook's Apache license covers the cookbook code, not the weights.

```bash
python3 scripts/cookbook.py download
python3 scripts/cookbook.py verify-model
python3 scripts/cookbook.py check-processor
```

`check-processor` tests both image and video preprocessing in the pinned container without loading model weights. It catches patch-size and argument-layout problems before a long model startup.

The download is pinned to revision `fa56146238f9fcd5ab591b7052b31e28efdca5c3`. It includes **all** model and tokenizer assets, not just safetensors. Verification reads the full checkpoint, so allow time for approximately 129 GiB of disk I/O. Do not continue after a checksum mismatch. An existing download can be reused by setting `model_dir`; rerun `download` to align its metadata with the pinned revision before verifying.

The revision's recorded weights/config/tokenizer checksums match the tested checkpoint; its README changed since the original download. See [provenance](docs/PROVENANCE.md).

**Preserve the FP8 PLE table and scales.** Do not convert its shards to BF16. The required lookup patch transports FP8 bytes on SM80 and then applies the original scales; naïve conversion changes semantics and doubles table storage.

## 4. Build the optional all-reduce extension

If `tuned_all_reduce` is `true`:

```bash
python3 scripts/cookbook.py build-kernel
```

This uses the pinned image's compiler and Torch ABI, targets SM80, and uses two compiler jobs. No local CUDA toolkit is needed. The output lives in `state_dir/extensions/`; `build.json` records its SHA-256, source hashes, image, Torch, and CUDA versions. No precompiled `.so` is distributed.

This extension uses the exact pinned vLLM communicator layout. A runtime guard checks that layout before dispatch. It is enabled only for contiguous BF16 tensors, four fully connected GPUs, and SM80. Rebuild **and revalidate** it after any image/compiler/Torch/GPU change. See the [kernel study](docs/CUSTOM_ALL_REDUCE.md).

## 5. Start and validate a manual deployment

First ensure another model server is not using the four GPUs or port 19088. The launcher refuses to replace a container that lacks its ownership label. Inspect the full launch command without starting anything:

```bash
python3 scripts/cookbook.py command
python3 scripts/cookbook.py start
python3 scripts/cookbook.py wait --timeout 900
```

Cold startup usually takes **4–8 minutes**, depending on host load and caches. In another terminal, follow logs (replace the name if you changed it):

```bash
podman logs -f qwen38-a100
```

Expected log indicators include MARLIN, filtering by the safetensors index, pinned CPU FP8 PLE storage, `Resolved Mamba cache block size: construction=16, config=800, spec=800`, and `Application startup complete`. With tuning enabled, expect `Using tuned SM80 BF16 all-reduce kernels.`

Check the API and a normal answer:

```bash
curl -fsS http://127.0.0.1:19088/health
curl -fsS http://127.0.0.1:19088/v1/models
curl -fsS http://127.0.0.1:19088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-uncensored","messages":[{"role":"user","content":"Return only the integer: 17 multiplied by 19."}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

Expected answer: `323`. The model ID is `qwen38-flash-next-uncensored`; client base URL is `http://127.0.0.1:19088/v1`. With `vision: true`, the same endpoint also accepts image and video attachments; see the [request format and visual checks](docs/VISION.md).

The API binds to loopback without authentication. To use it remotely, forward that port over SSH, or place an authenticated proxy in front of it. Keep the example's loopback binding for direct deployments.

Run regression checks:

```bash
mkdir -p results
podman exec qwen38-a100 python3 /cookbook/benchmarks/test_mamba_cache_block_size.py
python3 benchmarks/stress_prefix_cache.py --port 19088 --output results/cache-stress.json
python3 benchmarks/validate.py --port 19088 --output results/validation.json
```

Expected: **3 unit tests, 32/32 cache checks, 13/13 functional checks**. The functional suite covers thinking, changed long prefixes, a tool-call round trip, request cancellation, and concurrent request isolation. Nonzero exit means inspect the saved output before relying on the deployment.

## 6. Enable systemd startup and recovery

After manual validation:

```bash
python3 scripts/cookbook.py stop
python3 scripts/cookbook.py install-service
sudo loginctl enable-linger "$USER"
systemctl --user start qwen38-a100.service
python3 scripts/cookbook.py wait --timeout 900
systemctl --user status qwen38-a100.service
```

The installer writes and enables `~/.config/systemd/user/qwen38-a100.service`. It starts the **same pinned, patched configuration** via an attached Podman process. The unit records the exact Podman executable so a different systemd PATH cannot select another installation. `Restart=always` restarts it after unexpected container termination; an intentional `systemctl stop` leaves it stopped. Linger starts the user manager at boot without an interactive login.

Once systemd manages it, use:

```bash
systemctl --user restart qwen38-a100.service
python3 scripts/cookbook.py wait --timeout 900
journalctl --user -u qwen38-a100.service -f
```

The unit is `Type=simple`: `active` alone does not mean model loading finished. Always use the health/readiness check. A service restart recreates the container from the current checkout/config. Keep these paths intact. The launcher uses a non-inherited process lock so detached Podman processes cannot retain it.

If migrating from an earlier setup, disable its model-serving unit before enabling this one. The cookbook cannot identify all unrelated services on your machine. Do not have two units compete for the same GPUs. Restarting `podman.service` alone restarts Podman's API service, not this model service.

The packaging/service checks and original GPU experiments have different validation scopes; see [validation status](docs/VALIDATION.md).

## 7. Measure single and parallel throughput

Run benchmarks with other inference traffic idle:

```bash
python3 benchmarks/bench.py --port 19088 --tokens 128 --output results/bench-128.json
python3 benchmarks/bench.py --port 19088 --tokens 256 --wide --prefill --output results/bench-256.json
```

The benchmark warms relevant shapes, runs two single-request trials, two concurrent requests, and a longer prompt. `--wide` adds four and eight requests; `--prefill` adds a fresh approximately 8K prompt. Fixed-length performance requests use greedy decoding, thinking disabled, and `ignore_eos`; correctness requests terminate normally.

Original text-only tuned configuration, four repeated 256-token suites (the image/video-enabled quiet repeat is recorded in the multimodal guide):

| Simultaneous requests | Median decode tokens/s per request | Aggregate output tokens/s including startup |
|---:|---:|---:|
| 1 | ~108.5 | ~104.7 |
| 2 | ~103.8 | ~195.0 |
| 4 | ~99.7 | ~365.9 |
| 8 | ~84.6 | ~615.2 |

Decode rate excludes time to first token. Aggregate rate includes request startup, so it is not simply the per-request decode rate multiplied by concurrency. SSE can batch multiple tokens per chunk, especially with MTP; these are client-observed streaming estimates using API token counts. Compare the same prompts, output length, cache state, and load. Results are workload-specific, not a hardware-wide guarantee.

**Context limit:** the example configuration selects **524,288 total tokens with 2× YaRN**; the checkpoint's native limit is 262,144. This recipe does not enable 1M. [Test details and reproduction](docs/CONTEXT.md).

## 8. Verify a near-200K context with a real book

Prepare the complete public-domain *A Tale of Two Cities* fixture:

```bash
python3 scripts/prepare_book.py
```

This extracts the bundled, compressed original Project Gutenberg text, verifies its recorded hash, and produces the same clean text. The raw Gutenberg header/license is retained in the fixture and in `data/`. To check a new upstream download, use `python3 scripts/prepare_book.py --download`; a changed hash stops the check rather than silently changing the benchmark.

After a server restart, before any request with this book prefix:

```bash
python3 benchmarks/book_bench.py --port 19088 --output results/book-cold.json
python3 benchmarks/book_bench.py --port 19088 --output results/book-cached.json
```

The actual API prompt is **190,422 tokens**: the complete novel, four short margin notes placed across its length, and instructions. There is no synthetic bulk padding. The model is asked for an approximately 500-word summary, seven factual answers, all four note codes in order, and a short ending quotation.

| Book request | First-token latency | Decode rate | Output tokens |
|---|---:|---:|---:|
| Cold | 16.781 s | 108.87 tokens/s | 971 |
| Cached | 1.288 s | 108.62 tokens/s | 971 |

Fresh/cached answers and stock/tuned-kernel answers were byte-identical. All four codes and seven targeted plot answers were correct. The summary was coherent but had two inaccuracies: it claimed Manette believed Lucie dead, and attributed the prison letter to Madame rather than Ernest Defarge. Inspect the saved `.answer.md` yourself; passing marker checks does not establish factual accuracy. [Read the measured answer](measurements/2026-09-11/09-custom-ar-kernel-tuned/book.answer.md).

## What changed performance?

| Change | Representative measured effect |
|---|---|
| Decode CUDA graphs | 4.78 → 47.18 single-request tokens/s |
| MTP, one draft token | 47.18 → 75.37 |
| Automatic NCCL selection instead of forced Ring/Simple | 75.37 → 96.86 |
| SM80 dense GEMM tuning | Matched 256-token confirmation: 90.93 → 98.93 |
| More sequences and larger prefill budget | Eight-request throughput ~543 before later graph/collective improvements; ~8K TTFT 2.51 → 1.02 s |
| Corrected prefill graphs + existing custom all-reduce | ~107.83 single-request and ~608 aggregate tokens/s |
| Additional custom all-reduce tuning | Repeated stock control 107.785 → tuned 108.458 (+0.624%) |

These rows span different cohorts and settings; don't multiply their ratios. The final **4.78 → 107.88** comparison uses the same 128-token workload. [Full results](docs/RESULTS.md) distinguish output lengths and preserve rejected experiments.

MTP depths 2 and 3 were slower. Initial prefill graphs and custom all-reduce failed correctness because of a **shared recurrent-cache indexing bug**, not established reduction arithmetic corruption: construction cached block size 16 before the backend finalized it to 800. A reused 2,400-token prefix restored column 149 instead of column 2. The patch uses resolved `MambaSpec.block_size` or the finalized live configuration. Both settings passed after that fix.

## Troubleshooting and maintenance

| Symptom | What to check |
|---|---|
| Roughly 5 tokens/s | CUDA graphs may be disabled. Inspect `command`, actual container arguments and startup logs. |
| `Register Register` or corruption after prefix reuse | Ensure the Mamba cache overlay is mounted and its hash matches; run cache regressions. |
| Missing `down_proj` / `w2_weight` | Loader index filtering and both loader overlays must be enabled. |
| `fp8e4nv not supported` | The SM80 byte-preserving PLE overlay is missing. |
| Empty decoder prompt | Download all tokenizer assets; verify checkpoint checksums. |
| OOM during load/capture | Check other GPU users, host RAM, the exact image and AWQ/MARLIN logs. Avoid raising graph capture sizes blindly. |
| Extension import or ABI failure | Rebuild inside the pinned image; use `tuned_all_reduce: false` to return to the existing kernel. |
| Service active, API unavailable | Model loading takes minutes; use `wait` and logs. |
| No startup after reboot | Verify unit enablement and `loginctl show-user "$USER" -p Linger`. |
| Tool call appears as text | Use automatic tools with `qwen3_xml` and run the full round-trip check. |

To stop a managed deployment, use `systemctl --user stop qwen38-a100.service`; for manual mode, use `python3 scripts/cookbook.py stop`. To remove automatic startup, use `systemctl --user disable --now qwen38-a100.service`. Model files and caches are not deleted.

For changes, keep baseline results, change one setting at a time, restart, rerun correctness and matched performance tests, and retain improvements only. Do not upgrade the image independently of these overlays. The cookbook exposes path/name/port, vision, CPU-cache capacity, and optional-kernel settings intentionally; broader experiments require reviewing the launcher and repeating validation.

## Repository and publication

- [Enable and verify image/video input](docs/VISION.md)
- [CPU KV-cache configuration, verification and results](docs/CPU_CACHE.md)
- [Reproduce individual historical experiments](docs/EXPERIMENTS.md)
- [Source/patch provenance and license notes](docs/PROVENANCE.md)
- [Exact patch behavior and reconstruction](docs/PATCHES.md)
- [All-reduce build, numerical tests, and experiment limits](docs/CUSTOM_ALL_REDUCE.md)
- [Validation performed on this cookbook](docs/VALIDATION.md)
- [How to publish this prepared repository](docs/PUBLISHING.md)

Run `python3 -m unittest discover -s tests -v`, `python3 scripts/check_repository.py`, and `python3 scripts/verify_patches.py` before publishing changes. GitHub Actions runs these CPU-side checks; it does not claim to run GPU inference. No model weights, private server configuration, remote access credentials, compiler caches, or compiled extension binaries belong in this repository.
