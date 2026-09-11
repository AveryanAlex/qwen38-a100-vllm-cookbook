# Required patches

The launcher mounts the complete files in `overlays/` over the pinned image's Python package. It checks their hashes before starting; there is no wheel rebuild. These files retain vLLM copyright/SPDX headers. The exact source modifications are in `patches/`.

| Patch | Files | Purpose |
|---|---|---|
| 01-loader-ple-sm80.patch | default_loader.py, weight_utils.py, ngram_embedding.py | Filter safetensors by indexed tensor names, preserve the PLE bytes and scales, and make pinned CPU PLE lookup work on SM80. |
| 02-sm80-gemm.patch | low_latency_gemm.py, _skinny_gemm.py | Replace the SM90-only CuTe election operation with lane-zero selection where appropriate; dispatch 22 measured small-batch dense shapes to SM80 plans. |
| 03-mamba-prefix-cache.patch | mamba_hybrid.py | Use the finalized recurrent-cache block size when restoring a cached prefix. |
| 04-custom-ar-dispatch.patch | custom_all_reduce.py | Opt into the compiled tuned reduction extension on compatible SM80 BF16 TP4 operations; preserve the normal dispatcher otherwise. |
| 05-custom-ar-threshold.patch | custom_all_reduce.cuh | Extend one-stage reduction through 1 MiB inclusive in the optional extension's local header. |
| 06-offload-scratch-groups.patch | offloading_connector.py | Exclude non-cacheable scratch rings from the offloader and consistently map attention/Mamba group indices. |

Patches 01–03 are used in the main recipe. Patches 04–05 and `kernels/custom-ar/tuned.cu` implement the optional 0.62% extra tuning. Disabling it leaves the existing custom all-reduce enabled, with the shared cache fix still applied.

Patch 06 is required for this model's native CPU cache. Without it, enabling offloading fails during startup even with prefix caching enabled. See the [CPU cache runbook](CPU_CACHE.md).

## Reconstruct the overlays

The source basis is `wtdcode/vllm-backport`, commit `24cb31bb4fd0becee65c810c913a8caa4f610c36`. The image is a separately pinned immutable artifact; source-file identities for both are recorded in `manifest.json` rather than inferred from a tag.

```bash
python3 scripts/verify_patches.py
```

This downloads the eight original Python files at the pinned commit into a temporary directory, checks their hashes, applies patches 01–04 and 06, and compares the resulting bytes with every shipped overlay. It also verifies patch 05 against the shipped original/tuned CUDA headers. The temporary files are removed after verification. It does not change the checkout or installed vLLM.

To inspect the diffs manually, read the numbered `.patch` files. A historical loader/PLE patch omitted later PLE changes; patch 01 in this cookbook was regenerated from the pinned source to the final deployed file, and exact reconstruction was verified. Use this complete patch rather than an earlier standalone copy.

## Recurrent-cache failure

`MambaHybridModelState` was initialized before platform alignment finalized the cache block size. Its construction-time snapshot was 16. The backend later selected 800 with MTP enabled (784 in the tested non-MTP layout).

On prefix reuse, `add_request` calculated `(num_computed_tokens - 1) // block_size` using the stale snapshot. At 2,400 cached tokens it chose **149 instead of 2**, and generation restored the wrong recurrent state. Changes in allocation exposed this as `duct Register Register ...` with prefill graphs or custom all-reduce.

The patch resolves the block size from the authoritative `MambaSpec` if available, otherwise from the live finalized cache configuration. The regression covers 784/800, boundaries, first requests, and spec precedence. Runtime logging confirmed 16 → 800 on all four ranks. After the fix, both prefill graphs and custom all-reduce passed 32 repeated-prefix checks and the cold/cached book test.

The observed failure was not established as reduction arithmetic corruption. Do not remove numerical tests, but do not diagnose that output alone as proof of a faulty reduction kernel.

## GEMM constraints

The dense GEMM patch preserves FP32 accumulation and BF16 output. Only measured shapes use its dispatch table; other shapes retain the existing linear implementation. This is not a replacement for the AWQ MoE backend: routed expert weights still use MARLIN. The LM head did not qualify for the selected table.

The original CuTe implementation failed to compile on SM80 because `elect_one` required SM90+. All 252 attempts in the initial sweep failed at that guard. Compatibility was fixed before timing candidate configurations. The shorter 128-token end-to-end gain was only around 1.6%; the matched 256-token follow-up measured about 98.93 vs 90.93 tokens/s. See the raw GEMM measurements and the cohort notes in the results report.
