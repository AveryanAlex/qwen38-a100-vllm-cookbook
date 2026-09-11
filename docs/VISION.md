# Image and video input

The checkpoint is `Qwen4ExpForConditionalGeneration` and contains a Qwen3-VL-style vision tower: 27 layers, hidden width 1152, and 333 indexed vision tensors. Vision weights are excluded from AWQ quantization by the checkpoint configuration.

The initial optimized deployment and the original performance report used `--language-model-only`. They did not load the tower. The cookbook now exposes `vision` in config.json for both image and video input. New example configurations enable it; existing configurations without the field preserve text-only behavior. Set it explicitly:

```json
"vision": true
```

Restart the service/container after changing this value. For systemd deployments:

```bash
systemctl --user restart qwen38-a100.service
python3 scripts/cookbook.py wait --timeout 900
```

## Required processor settings

Removing `--language-model-only` alone is insufficient for this checkpoint. Its preprocessor configuration names a Qwen2-VL processor with **patch_size=14**, while `config.json` defines a vision tower with **patch_size=16**. The runtime selects the Qwen3-VL processor class but still reads the serialized image patch size.

A direct processor test with a 512 × 512 input produced patch vectors of width **1176** by default (`3 × 2 × 14 × 14`), while the tower requires **1536** (`3 × 2 × 16 × 16`). Explicitly setting `patch_size=16` gives the correct vectors.

The multimodal recipe uses:

```text
--limit-mm-per-prompt '{"image":999,"video":999}'
--media-io-kwargs '{"video":{"num_frames":128}}'
--mm-processor-kwargs '{"patch_size":16,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"images_kwargs":{"min_pixels":4096,"max_pixels":1048576},"max_frames":128,"fps":2,"cap_pixels_per_frame":true,"size":{"shortest_edge":4096,"longest_edge":8388608}}'
```

Both modality counts are 999, the backport's effectively unrestricted default. Per-image resolution is approximately one megapixel. The video media loader is configured for up to 128 decoded/sample frames, and the processor targets 2 fps with max_frames=128 and a total pixel budget of 8,388,608. Long clips are sampled, not exhaustively analyzed frame by frame. These are per-item processing budgets, not low request item-count caps. The 262,144-token context and GPU/host memory still limit the actual number and sizes of items that fit together.

Image and video pixel budgets have different meanings. The flat `size` sets the video budget; `images_kwargs` overrides the image min/max pixels. Do not put video size in `videos_kwargs`: this backport computes and forwards a flat video `size`, and Transformers rejects duplicate flat/nested size arguments. The direct processor preflight checks this before loading weights. Flat image min/max pixel fields can otherwise override video budget calculations in vLLM. Explicit images_kwargs also avoid a legacy processor-loading problem: a size-only dictionary was overwritten by serialized min/max pixel fields. A 2400 × 2400 image with the corrected settings produces a 64 × 64 patch grid, 4096 vectors of width 1536, and 1024 merged visual tokens. Model and serialized processor files remain read-only.

## Correct image normalization

The serialized preprocessor also uses older CLIP normalization: mean `[0.48145466, 0.4578275, 0.40821073]`, standard deviation `[0.26862954, 0.26130258, 0.27577711]`. The Qwen3-VL/Qwen3.5 processor configurations use `[0.5, 0.5, 0.5]` for both. The recipe explicitly selects the latter values.

An image A/B on the same running model changed only normalization. With legacy normalization, the added `RAVEN 6247` label was read as `raven627` or `RAVEN 027`. With mean/std 0.5, both prompts read `RAVEN 6247` correctly. The marked image and request prompts were identical. Processor references and raw A/B outputs are preserved with the vision measurements. This evidence supports the corrected processor for this checkpoint; it is not a broad OCR evaluation.

## Bound CPU thread pools

The first multimodal startup loaded the vision weights but failed while profiling dummy image prompts: each worker's tokenizer attempted to initialize a Rayon pool sized to the large host CPU count. Thread creation failed with `ThreadPoolBuildError` / `Resource temporarily unavailable`. A separate processor probe also encountered libgomp thread exhaustion without bounds.

The container task limit is explicitly raised from the default 2048 to **8192** with `--pids-limit 8192`. This can also be changed on a running container with `podman update --pids-limit 8192 <container>`. The measured retry retains bounded pools as well; the task-limit increase was applied during its startup and persisted for subsequent launches.

The vision recipe sets `TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`. These settings bound CPU pools without changing model weights or GPU precision. The original failed startup is recorded separately from successful measurements.

## Item counts versus actual capacity

`--limit-mm-per-prompt` accepts nonnegative integer counts: 0 disables a modality, and omitted modalities default to 999. There is no literal unlimited switch. The recipe explicitly sets both image and video to 999 so neither is silently disabled or capped at a small number.

This does not reserve enough memory for 999 full-resolution videos. The encoder scheduler, prompt context and available memory constrain actual requests. Increase per-item resolution/frame budgets only after checking memory and quality on your hardware. The tests below include three-image and three-video requests to verify that the original two-item restriction is gone; they do not claim to exhaustively test 999-item requests.

## Test with a classic photograph

The included NASA Apollo 8 **Earthrise** fixture was downloaded from Wikimedia Commons. It is a public-domain US government photograph. The raw file, exact derivatives, hashes and attribution are under `benchmarks/fixtures/vision/`.

```bash
mkdir -p results
python3 benchmarks/vision_bench.py --port 19088 --output results/vision.json
```

The script sends image bytes through OpenAI-compatible `image_url` data URLs. It does not include the filename, source URL or photograph title in the prompt. Inspect the saved responses against the actual image:

| Case | Expected visible evidence |
|---|---|
| Original | Gray rocky lunar terrain at bottom; blue/white Earth; black sky; no people/trees/spacecraft/printed labels. |
| Repeated original | Same observations after image/prefix reuse. |
| Vertically flipped | Gray rocky terrain at top, not bottom. |
| Lunar crop | Gray terrain; no blue/white Earth. |
| Marked | Exact text `RAVEN 6247`; red square at upper right; distinguish those graphics from the photo. |
| Two images | Earth appears in image 1; image 2 is a crop of the gray terrain without Earth. |
| Concurrent original/crop | Different answers reflect the correct image, with no cross-request image reuse. |

The script fails on API errors and preserves full responses. Semantic grounding requires review; recognizing the name Earthrise alone is not a pass.

## OpenAI-compatible request format

```json
{
  "model": "qwen38-flash-next-uncensored",
  "messages": [{
    "role": "user",
    "content": [
      {"type":"image_url","image_url":{"url":"data:image/png;base64,<encoded-image>"}},
      {"type":"text","text":"Describe only what is visible in this image."}
    ]
  }],
  "temperature": 0,
  "max_tokens": 600,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Use your client's normal image attachment support, or follow the benchmark's standard-library data-URL construction. The same endpoint and model alias accept plain text. The original text-only measurements remain a distinct cohort; vision-enabled throughput and quality results are recorded after validation.

## Verify video and mixed inputs

Install `ffmpeg` for preparing the external sample. The control clips are bundled; the real public OpenCV sample is downloaded on demand:

```bash
python3 scripts/prepare_video.py
python3 benchmarks/video_bench.py --port 19088 --output results/video.json
```

The test suite covers:

- A six-second control video: the printed words change **ALPHA → BRAVO → CHARLIE**, while the square changes **red/left → green/center → blue/right** over an Earthrise background.
- The reversed video, which must reverse the word/color/position order.
- A real six-second excerpt from OpenCV's vtest.avi: people walking through an outdoor road/intersection bordered by grass and buildings, with parked vehicles visible. No person identification is requested.
- Three videos in one request, requiring separate ordered descriptions.
- Three images in one request, including the original photo, Earth-free crop and labeled derivative.
- A mixed video-plus-image request, requiring the correct video sequence and the still-image label.

The video content part is a vLLM API extension; clients must support it or send the JSON directly. The script sends `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}` content parts through the same chat-completions endpoint. Video answers should reflect multiple frames; a plausible first-frame description alone does not pass the temporal checks. Audio understanding is not part of this test or recipe.

The bundled control clips are generated derivatives of the public-domain NASA photograph. Source, transformations and hashes are recorded in `benchmarks/fixtures/vision/video-source.json`. The real OpenCV sample is not redistributed in the cookbook; preparation verifies the source hash and extracts a silent six-second H.264 clip locally. Different ffmpeg versions can produce different encoded bytes for the same excerpt.

## Experiment history and rollout notes

- The initial image-enabled attempt loaded weights but exhausted CPU thread pools during tokenizer warmup at the default 2048-task container limit.
- Bounded pools and a task limit of 8192 allowed startup. Scene/flip/crop checks passed, but legacy normalization misread the added label.
- Corrected mean/std 0.5 fixed the label in controlled A/B prompts, and the eight image-grounding cases passed. Text/cache/book checks passed with the vision tower loaded.
- The first high-count image/video startup was interrupted before validation to separate image and video kwargs consistently. It is not a performance result.
- A nested `videos_kwargs.size` configuration failed video warmup because the backport also supplied flat `size`. The final recipe uses flat video `size` plus image-specific min/max overrides, and a direct processor check precedes another full startup.
- Stopping that process during GPU warmup exceeded Podman's first timeout. The container subsequently reached exited state; startup was retried only after checking that state. The launcher did not remove a running process after a failed stop.

Raw before/after answers, configuration records and final results are preserved with the multimodal measurements. These notes distinguish failed/interrupted attempts from measured results.

## Final production results (2026-09-11)

The production endpoint was verified with both modality counts set to **999**, the corrected processor flags, pinned image/source overlays, and an 8192-task limit. The exact launch settings and live-check results are in the multimodal measurement cohort.

- **8/8 targeted image-grounding cases:** original/repeated Earthrise, flipped terrain, Earth-free crop, exact RAVEN 6247 plus red square, two-image ordering, and concurrent image isolation.
- **6/6 targeted video/multi-input cases:** forward and reversed word/color/position order, real people walking outdoors, three videos, three images, and mixed video/image input.
- **13/13 functional checks** and **32/32 prefix-cache checks**.
- **190,422-token book** completed with all four reference codes and all seven targeted plot answers correct. Cold TTFT was 17.22 s; cached TTFT 1.58 s. These book requests overlapped other production traffic; decode was 99.49/94.77 tokens/s respectively. Fresh/cached wording differed in this shared-load run.

A quiet 256-token text suite after enabling image and video measured:

| Concurrent requests | Decode tokens/s per request | Aggregate tokens/s |
|---:|---:|---:|
| 1 | 108.42 | approximately 104.7 |
| 2 | 103.60 | 194.54 |
| 4 | 99.47 | 363.64 |
| 8 | 84.45 | 612.68 |

Every scenario's server generation-token delta exactly matched the benchmark's requested output count, with no additional request traffic detected. The earlier shared-load suite is also preserved and is not used to claim an isolated slowdown.

These are targeted checks, not perfect vision or timing accuracy. The model added incorrect timestamps (up to 7 s) to one reversed six-second clip, despite reporting the correct event order. It speculated incorrectly that the isolated lunar crop was Charon. Some incidental image details and the real-video pedestrian-direction summary were imprecise. The book summary retained factual inaccuracies, including the belief that Lucie was dead and saying Carton bribed rather than blackmailed Barsad. Full answers and explicit review notes are preserved.

See [multimodal measurements](../measurements/2026-09-11-multimodal), [final summary](../measurements/2026-09-11-multimodal/final-summary.json), [video answers](../measurements/2026-09-11-multimodal/16-image-video/video.json), and [image answers](../measurements/2026-09-11-multimodal/16-image-video/vision.json).
