"""Run inside the pinned image; output a binary built against its Torch ABI."""
import hashlib
import json
from pathlib import Path
import shutil
from torch.utils.cpp_extension import load
import torch

root = Path('/cookbook')
w = root / 'kernels/custom-ar'
module = load(name='q38_ar_tuned', sources=[str(w / 'tuned.cu')],
              extra_include_paths=[str(w)], extra_cuda_cflags=['-O3', '-lineinfo'], verbose=True)
out = Path('/state/extensions')
out.mkdir(exist_ok=True)
shutil.copyfile(module.__file__, out / 'q38_ar_tuned.so')
manifest = json.loads((root / 'manifest.json').read_text())
record = {'image': manifest['image'], 'torch': torch.__version__, 'cuda': torch.version.cuda,
          'sha256': hashlib.sha256((out / 'q38_ar_tuned.so').read_bytes()).hexdigest(),
          'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in w.iterdir() if p.suffix in ['.cu', '.cuh']}}
(out / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
