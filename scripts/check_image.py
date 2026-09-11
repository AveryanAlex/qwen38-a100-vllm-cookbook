"""Check source compatibility before mounting any overlays."""
import hashlib
import json
from pathlib import Path
import sys

root = Path('/cookbook')
manifest = json.loads((root / 'manifest.json').read_text())
package = Path('/usr/local/lib/python3.12/dist-packages')
failures = []
for name, row in manifest['overlays'].items():
    path = package / row['package_path']
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != row['image_sha256']:
        failures.append(name)
    compile((root / 'overlays' / name).read_text(), name, 'exec')
    print(name, actual, 'OK' if name not in failures else 'MISMATCH')
if failures:
    sys.exit('Image sources do not match the tested build: ' + ', '.join(failures))
