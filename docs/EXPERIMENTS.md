# Reproducing the historical experiments

The main README deploys the final validated recipe. This guide reconstructs earlier hypotheses for controlled comparisons. Some deliberately omit the correctness fix and produce bad output; they are debugging experiments, not recommended serving configurations.

`scripts/experiment_command.py` prints the recorded configuration as a Podman create command with your local paths and the pinned image. It does not execute it. Historical profile JSON files are audit records; the script translates their original overlay identifiers. Model weights and the three loader/PLE overlays remain the same.

## Settings by profile

| Profile | Output tokens | MTP depth | Max sequences | Prefill budget | GPU memory budget | Custom AR | Graph mode |
|---|---:|---:|---:|---:|---:|---|---|
| 00-baseline | 128 | 0 | 2 | 800 | 0.9 | off | NONE |
| 01-graphs | 128 | 0 | 2 | 800 | 0.9 | off | FULL_DECODE_ONLY |
| 02-mtp1 | 128 | 1 | 2 | 800 | 0.9 | off | FULL_DECODE_ONLY |
| 02-mtp2 | 128 | 2 | 2 | 800 | 0.9 | off | FULL_DECODE_ONLY |
| 02-mtp3 | 128 | 3 | 2 | 800 | 0.9 | off | FULL_DECODE_ONLY |
| 03-auto-nccl | 128 | 1 | 2 | 800 | 0.9 | off | FULL_DECODE_ONLY |
| 03-custom-auto | 128 | 1 | 2 | 800 | 0.9 | on | FULL_DECODE_ONLY |
| 03-custom-ring | 128 | 1 | 2 | 800 | 0.9 | on | FULL_DECODE_ONLY |
| 04-sm80-gemm | 128 | 1 | 2 | 800 | 0.9 | off | FULL_DECODE_ONLY |
| 05-batch2048 | 256 | 1 | 8 | 2048 | 0.9 | off | FULL_DECODE_ONLY |
| 05-batch4096 | 256 | 1 | 8 | 4096 | 0.9 | off | FULL_DECODE_ONLY |
| 05-scheduler-baseline | 256 | 1 | 2 | 800 | 0.9 | off | FULL_DECODE_ONLY |
| 05-seqs4 | 256 | 1 | 4 | 800 | 0.9 | off | FULL_DECODE_ONLY |
| 05-seqs8 | 256 | 1 | 8 | 800 | 0.9 | off | FULL_DECODE_ONLY |
| 06-prefill-graphs | 256 | 1 | 8 | 4096 | 0.85 | off | FULL_AND_PIECEWISE |
| 07-prefill-cache-fix | 256 | 1 | 8 | 4096 | 0.85 | off | FULL_AND_PIECEWISE |
| 08-custom-ar-cache-fix | 256 | 1 | 8 | 4096 | 0.85 | on | FULL_AND_PIECEWISE |
| 09-custom-ar-kernel-tuned | 256 | 1 | 8 | 4096 | 0.85 | on | FULL_AND_PIECEWISE |
| 10-stock-ar-confirm | 256 | 1 | 8 | 4096 | 0.85 | on | FULL_AND_PIECEWISE |

Capture sizes, environment overrides, and selected patches are recorded in each profile config.json. Output lengths differ between cohorts. The GEMM patched confirmation used 256 tokens; its result is saved as confirm-256.json.

## Run one profile serially

Stop the systemd service first, if installed. For a manual deployment, stop the container. Replace the example container name below with the one in your config. Do not run alongside any other GPU workload.

```bash
systemctl --user stop qwen38-a100.service
podman stop --ignore --time 60 qwen38-a100
podman rm --ignore qwen38-a100
python3 scripts/experiment_command.py 01-graphs > /tmp/qwen38-experiment.sh
cat /tmp/qwen38-experiment.sh
bash /tmp/qwen38-experiment.sh
podman start qwen38-a100
python3 scripts/cookbook.py wait --timeout 900
python3 benchmarks/bench.py --port 19088 --tokens 128 --output results/01-graphs-repeat.json
```

Create `results/` first. Inspect the generated command before executing it. For 256-token profiles use `--tokens 256 --wide --prefill`; run correctness checks separately. The experiment command is a manually managed container even if a systemd unit is installed but stopped. Never enable two managers at once.

To return to the final recipe, start the systemd service, or use `python3 scripts/cookbook.py start` in manual mode. Both replace this cookbook-owned container with the final configuration. Changes to the local config such as port/name/state paths are honored.

For custom AR tuning, compare profile09 with profile10 across four suites each. The local compiled extension must already exist for profile09. Keep fresh and cached book results separate. Timing repetitions should use distinct output filenames so previous evidence is preserved.

## Dense GEMM screening

The measured shape sweep is preserved in `benchmarks/tune_gemm.py` and the raw gemm-tuning JSON files. It needs an isolated GPU and the skinny-kernel overlay. Stop inference first; run a disposable container using the same pinned image:

```bash
cookbook_root="$PWD"
cookbook_state="$HOME/.local/share/qwen38-a100"
cookbook_image="$(python3 -c 'import json; print(json.load(open("manifest.json"))["image"])')"
mkdir -p "$cookbook_state/gemm"
podman run --rm --device nvidia.com/gpu=all --entrypoint python3 \
  -v "$cookbook_root:/cookbook:ro" -v "$cookbook_state:/state" \
  -v "$cookbook_root/overlays/_skinny_gemm.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/cute_dsl/_skinny_gemm.py:ro" \
  "$cookbook_image" /cookbook/benchmarks/tune_gemm.py --mtp 1 --output /state/gemm/timings.json
```

Adjust cookbook_state to your configuration. Kernel minima alone do not justify a serving patch; compare the full model after numerical validation.
