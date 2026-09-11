# Artifact identity and provenance

The full machine-specific experiment archive remains separate from this public cookbook. This repository includes selected measurement JSON and generated book answers, preserving numeric values and failure results. It excludes private hostnames, account paths, container logs with machine addresses, remote tunnels, orchestration process files, model weights and compiled binaries.

## Runtime

- Image: `docker.io/lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3`.
- Historic tag: `v0.13.0-sm80`, the fork's version label.
- Source: https://github.com/wtdcode/vllm-backport at `24cb31bb4fd0becee65c810c913a8caa4f610c36`.
- Overlay hashes in `manifest.json` distinguish original source, original image, and patched file identity.
- Code under `overlays/` and the CUDA headers is derived from Apache-2.0 vLLM/backport code. Original headers are retained; modifications are identified in NOTICE and the numbered patches.
- The fork did not have a root NOTICE file at the pinned source URL. The upstream Apache license is preserved in `licenses/` and the cookbook includes its own attribution NOTICE.

The final deployment's Python overlays are included byte-for-byte. The optional CUDA extension is rebuilt from the preserved source inside the pinned image, and its new binary hash is recorded locally. Compiler output bytes can differ with build paths; the cookbook does not demand that a locally rebuilt binary equal the historical binary hash.

## Model

Repository: `leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32`.

The cookbook downloads revision `fa56146238f9fcd5ab591b7052b31e28efdca5c3`. Its checksum entries were compared with the checkpoint manifest on the measurement machine. The only differing entry was README.md; weights, model configuration, tokenizer assets and other listed artifacts had the same checksums. The original local snapshot's commit was not recorded, so this is a checked equivalent for inference artifacts, not a claim to have recovered that missing revision metadata.

`measurements/2026-09-11/model-SHA256SUMS` is the public revision's checksum manifest. Its own SHA-256 is pinned in `manifest.json`. Verification checks that manifest's identity and reads every listed file. The model config, tokenizer config and LICENSE also have explicit identity hashes.

Model weights are governed by the model's Qwen Community 1.0 license, not this cookbook's Apache license. See the pinned model LICENSE on Hugging Face before downloading or serving.

## Measurement conventions

The recorded date is 2026-09-11. Hardware: four A100-SXM4-40GB with all-pairs NV4, dual EPYC 7H12, approximately 503 GiB host RAM, Ubuntu 22.04, driver 580.126.09.

Client throughput uses streamed output token counts and wall-clock timestamps. Decode rate excludes first-token latency; aggregate rate includes startup and completion for the whole request batch. Greedy fixed-length throughput requests disable thinking and set ignore_eos. Functional/book checks terminate normally. SSE chunks are not necessarily individual tokens, so client decode rates are estimates, especially with MTP.

Historical profile config.json files are audit metadata. Original source mount paths have been replaced with repository-relative identifiers. Use the cookbook launcher for the final recipe; these JSON files are not accepted as launcher configuration. Initial and repeat benchmark output lengths are recorded separately and must not be mixed when calculating speedups.

Kernel screening happened under possible production load; later interleaved microbenchmarks still had load sensitivity. The private full-model tuned/stock repeat comparison is the basis for retaining the additional kernel patch.

## Book

The Project Gutenberg source URL and raw/clean hashes are recorded in `book-source.json`. The original complete text, including its Project Gutenberg header/license, is bundled as a gzip fixture. The preparation script can also fetch the upstream URL explicitly. The complete clean book has 190,170 model tokens; four margin notes and instructions bring the API prompt to 190,422. Model-generated answers and review notes are included as evaluation evidence, including known factual inaccuracies.

## Exclusions and interrupted work

Failed correctness outputs are retained in benchmark JSON. Machine-dependent server logs and abandoned controller scripts are not a supported deployment interface and are excluded. The results report documents the initial overlapping-controller interruption, the SM80 compile failures, and the inherited-lock incident. Publication does not turn those failed attempts into supported recipes.

## Vision follow-up

The initial deployment used language-model-only mode. The later image-input experiment removes that setting, aligns the processor patch size with the vision tower, and corrects normalization and sets image/video item counts to 999, with per-item resolution/frame budgets. Its results are a separate cohort; the original text-only records remain unchanged. The Earthrise fixture is a NASA public-domain photograph downloaded from Wikimedia Commons; source details, hashes and test alterations accompany the images.
