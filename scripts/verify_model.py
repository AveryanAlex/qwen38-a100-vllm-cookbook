"""Verify the checkpoint checksum manifest and all of its files."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'manifest.json').read_text())
model = Path(sys.argv[1]).expanduser().resolve()

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

for name, expected in manifest['model_identity_sha256'].items():
    if digest(model / name) != expected:
        sys.exit(f'Identity mismatch: {name}')
rows = []
for line in (model / 'SHA256SUMS').read_text().splitlines():
    if not line.strip():
        continue
    expected, name = line.split(maxsplit=1)
    name = name.removeprefix('*')
    path = (model / name).resolve()
    if not path.is_relative_to(model) or len(expected) != 64:
        sys.exit('Invalid checkpoint checksum entry')
    rows.append((path, expected))
for path, expected in rows:
    if digest(path) != expected:
        sys.exit(f'Checksum mismatch: {path.name}')
    print('OK', path.name, flush=True)
print(f'Verified {len(rows)} checkpoint files.')
