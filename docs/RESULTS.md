# Qwen3.8 performance experiments — 2026-09-11

Hardware: 4 × A100-SXM4-40GB with NV4 links; dual EPYC 7H12; Ubuntu 22.04; NVIDIA driver 580.126.09.
Model: leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32, TP4 + EP4, Marlin W4A16 g32, BF16 KV cache, pinned FP8 PLE offload. Native 262,144-token context is preserved.
Image: docker.io/lazymio/vllm-backport:v0.13.0-sm80, pinned for production by digest. This is the backport version, not upstream vLLM 0.13.

Selected configuration: `09-custom-ar-kernel-tuned`. The original deployment passed live health, image/argument/overlay identity, and functional checks. Cookbook packaging verification is documented separately in VALIDATION.md.

## Measurements

Selected raw measurements are in `measurements/2026-09-11/<name>/` (relative to the repository root). Benchmark JSON includes individual requests and correctness outputs; config.json is sanitized historical audit metadata, not launcher input. Original machine-specific logs and commands are omitted from the public package. Two warmed single-request runs, concurrent requests, and a longer prompt were measured directly on loopback. Decode tps excludes time to first token; aggregate tps includes request startup. Fixed-length performance requests use greedy decoding, thinking off, and ignore_eos. Separate correctness requests terminate normally. These are workload-specific measurements, not universal throughput guarantees.

| Experiment | Output tokens/request | Single decode tps (median of two) | Two-request aggregate tps | Eight-request aggregate tps | Cold ~8K TTFT (s) | Checks |
|---|---:|---:|---:|---:|---:|---|
| 00-baseline | 128 | 4.78 | 9.29 | — | — | Pass |
| 01-graphs | 128 | 47.18 | 76.33 | — | — | Pass |
| 02-mtp1 | 128 | 75.37 | 110.08 | — | — | Pass |
| 02-mtp2 | 128 | 74.27 | 104.60 | — | — | Pass |
| 02-mtp3 | 128 | 57.28 | 85.02 | — | — | Pass |
| 03-auto-nccl | 128 | 96.86 | 133.16 | — | — | Pass |
| 03-custom-ring | 128 | 102.76 | 141.46 | — | — | **Fail: cached-output corruption** |
| 03-custom-auto | 128 | 103.92 | 141.07 | — | — | **Fail: cached-output corruption** |
| 04-sm80-gemm | 128 | 98.37 | 136.36 | — | — | Pass |
| 05-scheduler-baseline | 256 | 90.93 | 154.65 | 153.77 | 3.408 | Pass |
| 05-seqs4 | 256 | 99.00 | 164.80 | 299.01 | 2.533 | Pass |
| 05-seqs8 | 256 | 99.18 | 163.87 | 543.44 | 2.513 | Pass |
| 05-batch2048 | 256 | 98.97 | 165.36 | 545.58 | 1.373 | Pass |
| 05-batch4096 | 256 | 99.06 | 164.97 | 544.63 | 1.024 | Pass |
| 06-prefill-graphs | 256 | 99.10 | 181.36 | 591.92 | 1.009 | **Fail: cached-output corruption** |
| 07-prefill-cache-fix | 256 | 99.16 | 177.47 | 580.68 | 1.022 | Pass |
| 08-custom-ar-cache-fix | 256 | 107.83 | 193.44 | 608.08 | 1.015 | Pass |
| 09-custom-ar-kernel-tuned | 256 | 108.44 | 195.05 | 608.29 | 1.019 | Pass |
| 10-stock-ar-confirm | 256 | 107.80 | 194.00 | 607.56 | 1.018 | Pass |

The 128-token and 256-token runs are separate cohorts. For the SM80 patch, `04-sm80-gemm/confirm-256.json` is the matched 256-token confirmation: 98.93/98.92 single-request tps, versus 90.92/90.93 without the patch in `05-scheduler-baseline`. The shorter 128-token gain was only about 1.6%. `08-custom-ar-cache-fix/final-128.json` is the first validated matched 128-token comparison with the original baseline. The selected profile has a final-128.json from production verification as well.

Matched final 128-token run: **4.78 → 107.88 decode tokens/s** (about 22.6×), and **9.29 → 185.73 aggregate tokens/s** for two simultaneous requests. This comparison uses the same prompts and output length.

## Decisions and changes

- Enable CUDA graphs. Decode-only graphs gave the largest gain, approximately 4.78 → 47.18 tps. Full + piecewise graphs were retained only after fixing cached-state restoration.
- Keep MTP depth 1. Depths 2 and 3 were slower on the tested prompts. Speculative graph sizes were adjusted with depth.
- Remove forced NCCL Ring/Simple. Automatic selection improved decode to approximately 97 tps. Custom all-reduce was initially rejected for corrupted cached output, then successfully retested with the recurrent-cache fix.
- Keep the additional custom all-reduce tuning after four repeated 256-token suites: 107.785 → 108.458 single-request decode tokens/s versus a fresh stock control (+0.624%). See CUSTOM_ALL_REDUCE.md for the limited scope, exact extension/ABI requirements, microbenchmark results and numerical checks.
- Keep the SM80 dense-projection patch. Replace unsupported CuTe elect_one with lane-zero selection, and dispatch 22 measured small-batch shapes to SM80-tuned configurations. FP32 accumulation and BF16 output are preserved. Untuned shapes use the existing linear implementation. The LM head did not qualify for the selected table.
- Increase maximum sequences from 2 to 8, and prefill budget from 800 to 4096. These improve aggregate throughput and prompt latency while preserving single-request decode speed.
- Reserve memory for bounded prefill graph capture with gpu_memory_utilization=0.85. This changes cache capacity/headroom, not precision or maximum context. A full near-200K request was tested; eight simultaneous near-200K requests were not tested.

## Correctness bug and fix

`MambaHybridModelState` is created before platform cache alignment. It captured a block size of 16, while the final MambaSpec and live cache configuration used 800 with MTP enabled (784 without MTP in the tested layout). add_request used the stale value to choose the recurrent-state source column. A cached 2400-token prefix therefore selected column 149 instead of column 2. Unpopulated/wrong block-table entries caused corrupted generation; allocation-changing settings exposed the latent bug.
The patch resolves the block size from MambaSpec when available, otherwise from the live finalized cache configuration. Runtime logs confirm construction=16, config=800, spec=800 on all four ranks. A deterministic unit reproducer fails on the old code and passes with the patch. Regression tests cover both 784 and 800, first requests, exact boundaries, and precedence of the resolved spec.
The failure was not established as arithmetic corruption inside the custom all-reduce kernel. Both prefill graphs and custom all-reduce passed after the shared cache-indexing fix.

## Near-200K book test

The complete public-domain A Tale of Two Cities was downloaded from https://www.gutenberg.org/ebooks/98.txt.utf-8. The clean book is 190,170 model tokens. Four margin-note reference codes and the task instructions bring the actual API prompt to 190,422 tokens. No synthetic bulk padding was used. The task requests a ~500-word summary, seven plot answers, all four reference codes, and a short ending quotation.

| Configuration | Cache | Input tokens | TTFT (s) | Decode tps | Output tokens | Markers |
|---|---|---:|---:|---:|---:|---|
| 05-scheduler-baseline | cold | 190422 | 63.336 | 94.98 | 1033 | 4/4, ordered |
| 05-batch4096 | cold | 190422 | 16.728 | 99.86 | 1039 | 4/4, ordered |
| 06-prefill-graphs | cold | 190422 | 16.563 | 99.45 | 1071 | 4/4, ordered |
| 07-prefill-cache-fix | cold | 190422 | 16.977 | 99.93 | 1071 | 4/4, ordered |
| 07-prefill-cache-fix | reused | 190422 | 1.327 | 99.71 | 1071 | 4/4, ordered |
| 08-custom-ar-cache-fix | cold | 190422 | 16.777 | 108.06 | 971 | 4/4, ordered |
| 08-custom-ar-cache-fix | reused | 190422 | 1.338 | 108.16 | 971 | 4/4, ordered |
| 09-custom-ar-kernel-tuned | cold | 190422 | 16.781 | 108.87 | 971 | 4/4, ordered |
| 09-custom-ar-kernel-tuned | reused | 190422 | 1.288 | 108.62 | 971 | 4/4, ordered |

Fresh and cached answers were byte-identical after the cache fix. The final candidate passed 32/32 repeated-prefix checks across eight distinct concurrent documents, 13/13 functional checks (thinking, long-prefix changes, tools, cancellation, request isolation), and the unit regression tests.
Manual book review: the main plot, ending, all seven targeted factual answers, and all four reference codes were correct. The summaries were coherent but not fully faithful: examples include the unsupported claim that Manette believed Lucie dead and attribution of the prison letter to Madame rather than Ernest Defarge. Raw answers and review notes are preserved; this is not a general model-quality evaluation.

## Failed and interrupted experiments

- Custom Ring, custom automatic NCCL, and initial prefill graphs produced “duct Register Register …” on a resumed medium prompt. These results were rejected before the cache fix; their raw outputs are retained.
- The original CuTe skinny kernel could not compile on SM80 because elect_one requires SM90+. All 252 attempted configurations failed at that guard. `measurements/2026-09-11/gemm-tuning.json.unsupported` preserves the compiler errors.
- An experiment-controller handoff overlapped model loading and interrupted a deployment attempt. `05-initial-aborted` contains no usable throughput measurement. A process lock was added to run.py; completed deployment/benchmark runs were subsequently serialized. Early microbenchmarks were used to select candidates; completed full-model A/B tests determined retention.
- The production shell launcher initially let conmon inherit flock descriptor 9, blocking the next serialized experiment after the launcher exited. Closing descriptor 9 in every Podman child fixed this. The final deployment explicitly checked the flock return code and returned 0. The cookbook includes a regression test for descriptor inheritance and lock release. The earlier lock check had mistakenly masked its failure with a later successful shell command.
- Old orchestration scripts are excluded from the cookbook. Use the serial commands in README.md; do not benchmark concurrently with another controller.

## Public artifacts

- `measurements/2026-09-11/`: performance suites, repeats, failures, quality checks and generated book answers.
- `overlays/`, `patches/`: complete runtime Python files and reconstructible patches.
- `kernels/custom-ar/`: extension source and numerical/timing probes.
- `benchmarks/`: portable reproduction clients and the cache unit regression.
- `manifest.json`: runtime/source/model identity and overlay hashes.
- README.md: current setup, service, benchmark and rollback commands.

Paths in the prose above identify original artifact basenames. The public package excludes machine-specific controller logs and uses the portable cookbook launcher instead. The additional all-reduce study is in CUSTOM_ALL_REDUCE.md.
