#!/usr/bin/env python3
"""Fetch pinned original files, apply the patches, and compare overlay bytes."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

root = Path(__file__).resolve().parents[1]
m = json.loads((root / 'manifest.json').read_text())
with tempfile.TemporaryDirectory(prefix='qwen38-patch-check-') as directory:
    work = Path(directory)
    for name, row in m['overlays'].items():
        path = row['package_path']
        url = f'https://raw.githubusercontent.com/wtdcode/vllm-backport/{m["source_commit"]}/{path}'
        original = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(original).hexdigest() != row['upstream_sha256']:
            raise SystemExit(f'Upstream hash mismatch: {name}')
        dest = work / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(original)
    for patch in sorted(p for p in (root / 'patches').glob('*.patch') if p.name != '05-custom-ar-threshold.patch'):
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=work, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=work, check=True)
    for name, row in m['overlays'].items():
        if (work / row['package_path']).read_bytes() != (root / 'overlays' / name).read_bytes():
            raise SystemExit(f'Patched file does not match overlay: {name}')
    path = work / 'csrc/custom_all_reduce.cuh'
    path.parent.mkdir(exist_ok=True)
    path.write_bytes((root / 'kernels/custom-ar/custom_all_reduce.cuh').read_bytes())
    subprocess.run(['git', 'apply', str(root / 'patches/05-custom-ar-threshold.patch')], cwd=work, check=True)
    if path.read_bytes() != (root / 'kernels/custom-ar/custom_all_reduce_tuned.cuh').read_bytes():
        raise SystemExit('Kernel threshold patch mismatch')
print('All patches reproduce the shipped overlays and tuned kernel header exactly.')
