#!/usr/bin/env python3
"""Offline release checks: hashes, syntax, local links and private-path leaks."""
import gzip
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'manifest.json').read_text())
for name, row in manifest['overlays'].items():
    assert hashlib.sha256((root / 'overlays' / name).read_bytes()).hexdigest() == row['patched_sha256'], name
for line in (root / 'measurements/SHA256SUMS').read_text().splitlines():
    expected, relative = line.split('  ', 1)
    assert hashlib.sha256((root / 'measurements' / relative).read_bytes()).hexdigest() == expected, relative
fixtures = root / 'benchmarks/fixtures'
for name, source in json.loads((fixtures / 'extended-context-sources.json').read_text()).items():
    raw = gzip.decompress((fixtures / (name + '.txt.gz')).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == source['raw_sha256'], name
    assert len(raw) == source['raw_bytes'], name
ignored = {'.git', '__pycache__', 'data', 'results', 'state', '.venv'}
for p in root.rglob('*'):
    if not p.is_file() or any(x in ignored for x in p.relative_to(root).parts) or p.name == 'config.json':
        continue
    if p.suffix == '.py':
        compile(p.read_text(), str(p), 'exec')
    if p.suffix not in ['.py', '.json', '.md', '.patch', '.cu', '.cuh', '.txt', '.yml']:
        continue
    text = p.read_text()
    # Match concrete personal machine paths, not generic examples.
    assert not re.search(r'/home/(?:alex|averyan)/|serv1\.asc\.|(?:87\.120\.93\.107|194\.238\.79\.211)', text), p
    if p.suffix == '.md':
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' in target or target.startswith('#'):
                continue
            target = target.split('#')[0]
            assert (p.parent / target).exists(), (p, target)
print('Repository syntax, overlay/measurement/book identities, local links and private-path checks passed.')
