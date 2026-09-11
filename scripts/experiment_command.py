#!/usr/bin/env python3
"""Print a historical Podman create command using local cookbook paths."""
import argparse
import json
from pathlib import Path
import shlex
from cookbook import ROOT, config, create_command, MANIFEST

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('profile', help='Recorded profile name, e.g. 01-graphs')
p.add_argument('--config', type=Path, default=ROOT / 'config.json')
a = p.parse_args()
profiles = ROOT / 'measurements/2026-09-11'
# Profile must be a single recorded directory name, not an arbitrary path.
if a.profile not in {d.name for d in profiles.iterdir() if d.is_dir()}:
    p.error('Unknown recorded profile')
historical = json.loads((profiles / a.profile / 'config.json').read_text())
c = config(a.config)
# Replaying an older experiment must not inherit newer serving defaults.
c['max_model_len'] = historical.get('max_model_len', 262144)
c['kv_offloading_gib'] = historical.get('kv_offloading_gib', 0)
c['vision'] = historical.get('vision', False)
c['tuned_all_reduce'] = historical.get('env', {}).get('QWEN_SM80_AR_TUNING') == '1'
args = create_command(c)
# Preserve the historical API-usage setting as well as the inference settings.
args.remove('--enable-prompt-tokens-details')
# Retain just the three model-loading/PLE overlays plus historical performance overlays.
mapping = {'low_latency_gemm_sm80.py': 'low_latency_gemm.py',
           '_skinny_gemm_sm80.py': '_skinny_gemm.py',
           'mamba_hybrid_cache_fix.py': 'mamba_hybrid.py',
           'custom_all_reduce_tuned.py': 'custom_all_reduce.py'}
selected = {'default_loader.py', 'weight_utils.py', 'ngram_embedding.py'}
selected.update(mapping[Path(k).name] for k in historical.get('mounts', {}) if Path(k).name in mapping)
i = 0
while i < len(args):
    if args[i] == '--volume':
        src = args[i + 1].split(':', 1)[0]
        if Path(src).parent == ROOT / 'overlays' and Path(src).name not in selected:
            del args[i:i + 2]
            continue
    i += 1
for flag, value in [('--max-num-seqs', historical.get('seqs', 2)),
                    ('--max-num-batched-tokens', historical.get('batch', 800)),
                    ('--gpu-memory-utilization', historical.get('memory', 0.9)),
                    ('--compilation-config', json.dumps(historical.get('compilation', {'mode': 0, 'cudagraph_mode': 'NONE'})))]:
    args[args.index(flag) + 1] = str(value)
i = args.index('--speculative-config')
if historical.get('mtp'):
    args[i + 1] = json.dumps({'method': 'mtp', 'num_speculative_tokens': historical['mtp']})
else:
    del args[i:i + 2]
if historical.get('disable_custom', True):
    args += ['--disable-custom-all-reduce']
env = {'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'Simple'}
env.update(historical.get('env', {}))
for key in ['NCCL_ALGO', 'NCCL_PROTO']:
    if env[key] is not None:
        args[2:2] = ['--env', f'{key}={env[key]}']
extras = historical.get('extra', [])
if historical.get('vision'):
    # Replace cookbook defaults with the exact historical modality settings.
    for flag in ['--limit-mm-per-prompt', '--mm-processor-kwargs', '--media-io-kwargs']:
        if flag in args:
            i = args.index(flag)
            del args[i:i + 2]
args += extras
print(shlex.join(args))
