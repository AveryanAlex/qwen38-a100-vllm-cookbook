#!/usr/bin/env python3
"""Prepare the exact book fixture used by the 190K-token experiment."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--download', action='store_true', help='Fetch upstream instead of using the bundled original')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
data = root / 'data'
data.mkdir(exist_ok=True)
reference = json.loads((root / 'measurements/2026-09-11/book-source.json').read_text())
raw = data / 'tale-of-two-cities.txt'
if a.download:
    with urllib.request.urlopen(reference['url'], timeout=90) as response:
        raw.write_bytes(response.read())
else:
    raw.write_bytes(gzip.decompress((root / 'benchmarks/fixtures/tale-of-two-cities.txt.gz').read_bytes()))
if hashlib.sha256(raw.read_bytes()).hexdigest() != reference['raw_sha256']:
    raise SystemExit('The upstream book changed; use the original fixture or record a new benchmark cohort.')
text = raw.read_text()
text = re.split(r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n', text, maxsplit=1)[-1]
text = re.split(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK', text, maxsplit=1)[0].strip()
clean = data / 'tale-of-two-cities-clean.txt'
clean.write_text(text)
if hashlib.sha256(clean.read_bytes()).hexdigest() != reference['clean_sha256']:
    raise SystemExit('Clean book fixture hash mismatch')
print('Prepared exact book fixture. Expected clean tokens: 190170; API prompt tokens: 190422.')
