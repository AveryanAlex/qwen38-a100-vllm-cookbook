# Additional custom all-reduce tuning

The main recipe already reached approximately 108 tokens/s using the fork's existing custom all-reduce after fixing recurrent-cache indexing. This follow-up changed CUDA launch geometry and found a modest extra gain. It is optional because it adds an exact-image ABI dependency.

## Kernel and scope

The original communicator uses a fixed 512 threads/block, at most 36 blocks, and one-stage reduction below 512 KiB. The tuned extension preserves the original class layout, allocation, buffer registration, CUDA graph metadata, cleanup, all-gather and reduce-scatter behavior.

The wrapper calls the pinned header's inline `CustomAllreduce::allreduce` with a shape-dependent thread count. Before using its pointer, an ABI guard checks rank/world, signal pointers, self-signal, rank-data bounds and the registered buffer map. This is a consistency guard for the pinned image, not a general ABI compatibility guarantee.

Dispatch requires `QWEN_SM80_AR_TUNING=1`, world size 4, fully connected peers, SM80 and contiguous BF16 input/output. Other paths retain the original dispatcher. The block limit stays 36:

| Upper payload size (bytes) | Threads/block |
|---:|---:|
| 10,240 | 64 |
| 40,960 | 128 |
| 81,920 | 256 |
| 163,840 | 128 |
| 327,680 | 256 |
| 1,048,576 | 128 |
| Larger | 512 |

One-stage reduction is used through 1 MiB inclusive; larger messages retain two-stage reduction. The threshold patch only affects the optional extension's header. Rebuild and revalidate before changing image, compiler, Torch, GPU or topology.

## Screening and numerical checks

352 configurations were screened: two algorithms, four thread counts, four block limits, and eleven payload sizes from 5 KiB to 4 MiB. Integer reference checks passed. Screening overlapped production traffic, so minimum/absolute timings were candidate-selection evidence only.

Follow-up timing interleaved stock and tuned kernels in ABBA order. Tests mutated inputs between CUDA graph replays and alternated kernels sharing synchronization state. Integer sums were checked exactly. Twelve random BF16 cases (three sizes, four seeds) were compared with FP32-sum/BF16-output references at `rtol=1/128, atol=.001`; small one-stage stock/tuned outputs also matched exactly. All four ranks participated. Follow-up microtimings remained load-sensitive, especially the largest message.

| Payload | Stock µs | Tuned µs |
|---|---:|---:|
| 5 KiB | 5.530 | 3.942 |
| 10 KiB | 5.683 | 3.994 |
| 20 KiB | 5.734 | 4.147 |
| 40 KiB | 5.939 | 4.506 |
| 80 KiB | 6.554 | 5.734 |
| 160 KiB | 7.782 | 7.066 |
| 320 KiB | 10.291 | 9.421 |
| 512 KiB | 15.309 | 11.725 |
| 1 MiB | 20.582 | 18.432 |
| 2 MiB | 26.982 | 26.957 |
| 4 MiB, load affected | 227.251 | 228.070 |

Raw data is in `measurements/2026-09-11/ar-*.json`. The initial probe build failed when `ATen/cuda/CUDAContext.h` pulled unavailable cusparse headers. Using `c10/cuda/CUDAStream.h` fixed compilation without installing extra packages.

## Full-model decision

Private full-model tests kept prompts, 256 output tokens and other configuration identical. Four suites per configuration each contained two single-request trials and two/four/eight-request batches. A fresh stock control followed the tuned run. These are repeated sequential tests, not randomized independent trials.

| Configuration | Median of suite single-request rates | Suite range | Eight-request aggregate median |
|---|---:|---:|---:|
| Stock custom all-reduce, fresh control | 107.785 tokens/s | 107.732–107.799 | 611.45 tokens/s |
| Tuned extension | 108.458 tokens/s | 108.427–108.549 | 615.19 tokens/s |

All eight individual tuned single-request trials exceeded all eight stock trials. The **0.624%** single-request improvement was retained. The aggregate difference is more scheduling-sensitive and should not be treated as universal. The earlier stock profile measured 107.831 tokens/s, consistent with the new control.

The tuned full model passed 3 cache unit tests, 32/32 cache stress cases, 13/13 functional cases, and the 190,422-token book test. Fresh/cached and stock/tuned book answers were byte-identical. Cold book TTFT remained effectively unchanged at 16.78 seconds; decode was 108.87 tokens/s. Four markers and seven targeted factual answers were correct; the summary still had the documented two inaccuracies.

## Build and run numerical probes

Build the deployable extension using the command in README.md:

```bash
python3 scripts/cookbook.py build-kernel
```

For microbenchmarks, first stop model serving so it does not affect timings. These commands use a **separate disposable GPU container**, the same pinned image, and the stock Python communicator; they do not start the full model. Set `cookbook_state` to the absolute `state_dir` used in config.json:

```bash
cookbook_root="$PWD"
cookbook_state="$HOME/.local/share/qwen38-a100"
cookbook_image="$(python3 -c 'import json; print(json.load(open("manifest.json"))["image"])')"
mkdir -p "$cookbook_state/ar" "$cookbook_state/cache"

podman run --rm --device nvidia.com/gpu=all --ipc=host \
  --security-opt seccomp=unconfined --entrypoint torchrun \
  -e QWEN_SM80_AR_TUNING=0 -e TORCH_CUDA_ARCH_LIST=8.0 -e MAX_JOBS=2 \
  -v "$cookbook_root:/cookbook:ro" -v "$cookbook_state:/state" \
  -v "$cookbook_state/cache:/root/.cache" \
  "$cookbook_image" --standalone --nproc-per-node=4 \
  /cookbook/kernels/custom-ar/verify_tuned.py
```

The output goes to `state_dir/ar/verified-timings.json` and `random-correctness.json`. To reproduce the full screening sweep, substitute `/cookbook/kernels/custom-ar/bench_ar.py`; it writes `kernel-timings.json`. Archive existing outputs first. Keep production stopped until the probe finishes, then start the service and wait for health again.

For full-model A/B, run the README's matched 256-token suite several times with `tuned_all_reduce: true`, then set it to `false`, restart and repeat with new output filenames. To measure the book cold, restart before its first request. Keep the recurrent-cache patch in both configurations.

The kernel probe scripts differ from the original experiment only in their writable output path: the cookbook is mounted read-only and results go to `/state/ar`. The CUDA implementation and its runtime overlay match the measured deployment.
