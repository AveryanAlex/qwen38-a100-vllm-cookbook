#!/usr/bin/env python3
"""Portable Podman launcher for the pinned, patched Qwen3.8 SM80 runtime."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'manifest.json').read_text())
IMAGE = MANIFEST['image']
LABEL = 'io.qwen38-a100-cookbook.managed'
SERVICE = 'qwen38-a100.service'
PODMAN = os.environ.get('QWEN_COOKBOOK_PODMAN') or shutil.which('podman') or 'podman'
MODEL_ALIAS = 'qwen38-flash-next-uncensored'
COMPILATION = {'mode': 0, 'cudagraph_mode': 'FULL_AND_PIECEWISE',
               'cudagraph_capture_sizes': [2, 4, 6, 8, 12, 16, 64, 256, 784, 4096],
               'max_cudagraph_capture_size': 4096}


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def config(path):
    c = json.loads(path.read_text())
    c.setdefault('vision', False)  # Preserve existing text-only configurations.
    allowed = {'model_dir', 'state_dir', 'container_name', 'port', 'tuned_all_reduce', 'vision'}
    if set(c) != allowed:
        raise ValueError(f'Config must contain exactly {sorted(allowed)}')
    for key in ['model_dir', 'state_dir']:
        p = Path(c[key]).expanduser()
        if not p.is_absolute():
            raise ValueError(f'{key} must be absolute (~/ is accepted)')
        c[key] = str(p.resolve())
        if any(ch in c[key] for ch in ':\n\r'):
            raise ValueError(f'{key} cannot contain colon or newline')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', c['container_name']):
        raise ValueError('Invalid container_name')
    if type(c['port']) is not int or not 1024 <= c['port'] <= 65535:
        raise ValueError('port must be an integer in 1024..65535')
    if type(c['vision']) is not bool:
        raise ValueError('vision must be boolean')
    if type(c['tuned_all_reduce']) is not bool:
        raise ValueError('tuned_all_reduce must be boolean')
    if any(ch in str(ROOT) for ch in ':\n\r'):
        raise ValueError('Repository path cannot contain colon or newline')
    return c


def state(c):
    return Path(c['state_dir'])


def check_sources():
    for name, info in MANIFEST['overlays'].items():
        actual = hashlib.sha256((ROOT / 'overlays' / name).read_bytes()).hexdigest()
        if actual != info['patched_sha256']:
            raise ValueError(f'Overlay hash mismatch: {name}')


def common(c):
    return [PODMAN, 'run', '--rm', '--entrypoint', 'python3',
            '--volume', f'{ROOT}:/cookbook:ro',
            '--volume', f'{state(c)}:/state']


def create_command(c):
    args = [PODMAN, 'create', '--name', c['container_name'], '--label', LABEL + '=true',
            '--device', 'nvidia.com/gpu=all', '--network', 'host', '--ipc', 'host',
            '--pids-limit', '8192',
            '--cap-add', 'SYS_PTRACE', '--security-opt', 'seccomp=unconfined']
    for value in ['VLLM_PLE_CPU_OFFLOAD=1', 'VLLM_ALLOW_LONG_MAX_MODEL_LEN=1',
                  'VLLM_FILTER_SAFETENSORS_BY_INDEX=1']:
        args += ['--env', value]
    if c.get('vision', False):
        for value in ['TOKENIZERS_PARALLELISM=false', 'RAYON_NUM_THREADS=1',
                      'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1']:
            args += ['--env', value]
    args += ['--volume', f'{c["model_dir"]}:/model:ro',
             '--volume', f'{state(c) / "cache"}:/root/.cache',
             '--volume', f'{state(c) / "triton"}:/root/.triton',
             '--volume', f'{ROOT}:/cookbook:ro']
    for name, info in MANIFEST['overlays'].items():
        if name == 'custom_all_reduce.py' and not c['tuned_all_reduce']:
            continue
        args += ['--volume', f'{ROOT / "overlays" / name}:/usr/local/lib/python3.12/dist-packages/{info["package_path"]}:ro']
    if c['tuned_all_reduce']:
        args += ['--env', 'QWEN_SM80_AR_TUNING=1', '--env', 'PYTHONPATH=/opt/q38-ar',
                 '--volume', f'{state(c) / "extensions"}:/opt/q38-ar:ro']
    args += [IMAGE, '/model', '--host', '127.0.0.1', '--port', str(c['port']),
             '--served-model-name', MODEL_ALIAS, '--tensor-parallel-size', '4',
             '--enable-expert-parallel', '--max-model-len', '262144',
             '--max-num-seqs', '8', '--max-num-batched-tokens', '4096',
             '--gpu-memory-utilization', '0.85',
             '--enable-prefix-caching', '--enable-auto-tool-choice',
             '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3',
             '--compilation-config', json.dumps(COMPILATION),
             '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 1})]
    if c.get('vision', False):
        args += ['--limit-mm-per-prompt', json.dumps({'image': 999, 'video': 999}),
                 '--mm-processor-kwargs', json.dumps({'patch_size': 16, 'images_kwargs': {'min_pixels': 4096, 'max_pixels': 1048576},
                                                     'image_mean': [0.5, 0.5, 0.5], 'image_std': [0.5, 0.5, 0.5],
                                                     'max_frames': 128, 'fps': 2, 'cap_pixels_per_frame': True,
                                                     'size': {'shortest_edge': 4096, 'longest_edge': 8388608}}),
                 '--media-io-kwargs', json.dumps({'video': {'num_frames': 128}})]
    else:
        args.insert(args.index('--enable-prefix-caching'), '--language-model-only')
    return args


def inspect(c):
    p = subprocess.run([PODMAN, 'container', 'exists', c['container_name']])
    if p.returncode == 1:
        return None
    p.check_returncode()
    return json.loads(subprocess.check_output([PODMAN, 'inspect', c['container_name']]))[0]


def owned(c):
    item = inspect(c)
    if item and item['Config']['Labels'].get(LABEL) != 'true':
        raise ValueError('Container name belongs to another deployment; choose a different name')
    return item


def unit_active():
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', SERVICE]).returncode == 0


def preflight(c):
    check_sources()
    for f in ['config.json', 'model.safetensors.index.json', 'tokenizer.json',
              'tokenizer_config.json', 'SHA256SUMS']:
        if not (Path(c['model_dir']) / f).is_file():
            raise ValueError(f'Model file missing: {f}; download and verify the full checkpoint')
    for d in ['cache', 'triton', 'extensions']:
        (state(c) / d).mkdir(parents=True, exist_ok=True)
    if c['tuned_all_reduce']:
        binary = state(c) / 'extensions/q38_ar_tuned.so'
        record = json.loads((state(c) / 'extensions/build.json').read_text())
        if record['image'] != IMAGE or hashlib.sha256(binary.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Extension identity mismatch; run build-kernel in the pinned image')
        source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (ROOT / 'kernels/custom-ar').iterdir()
                         if p.suffix in ['.cu', '.cuh']}
        if record['source_sha256'] != source_hashes:
            raise ValueError('Extension sources changed; run build-kernel again')


def start(c, foreground=False):
    if not foreground and unit_active():
        raise ValueError(f'Use systemctl --user restart {SERVICE} while the service is active')
    preflight(c)
    # Python file descriptors are non-inheritable; conmon cannot retain this lock.
    with (state(c) / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        item = owned(c)
        if item:
            if item['State']['Running']:
                run([PODMAN, 'stop', '--time', '60', c['container_name']])
            run([PODMAN, 'rm', c['container_name']])
        args = create_command(c)
        run(args)
        (state(c) / 'launch.json').write_text(json.dumps(args, indent=2) + '\n')
        if not foreground:
            run([PODMAN, 'start', c['container_name']])
    if foreground:
        # Exec keeps systemd's MainPID on the attached Podman client. ExecStop
        # stops the container; Restart=always handles unexpected termination.
        os.execvp(PODMAN, [PODMAN, 'start', '--attach', c['container_name']])


def stop(c):
    if unit_active():
        raise ValueError(f'Use systemctl --user stop {SERVICE} while the service is active')
    item = owned(c)
    if item and item['State']['Running']:
        run([PODMAN, 'stop', '--time', '60', c['container_name']])


def wait(c, timeout):
    deadline = time.monotonic() + timeout
    url = f'http://127.0.0.1:{c["port"]}'
    while time.monotonic() < deadline:
        item = inspect(c)
        if item and not item['State']['Running']:
            raise RuntimeError('Container exited; inspect podman logs')
        try:
            with urllib.request.urlopen(url + '/health', timeout=5) as r:
                if r.status != 200:
                    raise RuntimeError('Health check failed')
            with urllib.request.urlopen(url + '/v1/models', timeout=5) as r:
                models = json.load(r)['data']
            if not any(m['id'] == MODEL_ALIAS and m['max_model_len'] == 262144 for m in models):
                raise RuntimeError('Unexpected model alias or context length')
            print(f'Ready: {url}/v1 — {MODEL_ALIAS}, context 262144')
            return
        except (OSError, ValueError):
            time.sleep(15)
    raise RuntimeError('Startup timeout; inspect podman logs')


def unit_quote(value, command=True):
    # systemd specifier escaping applies to both directives. Dollar expansion
    # applies to ExecStart/ExecStop, not Environment assignments.
    escaped = str(value).replace(chr(92), chr(92) * 2).replace('"', chr(92) + '"').replace('%', '%%')
    if command:
        escaped = escaped.replace('$', '$$')
    return '"' + escaped + '"'


def service_text(c, config_path):
    return f'''[Unit]
Description=Qwen3.8 Flash Next on four A100 GPUs
StartLimitIntervalSec=0

[Service]
Type=simple
Environment={unit_quote("QWEN_COOKBOOK_PODMAN=" + PODMAN, command=False)}
ExecStart={unit_quote(sys.executable)} {unit_quote(ROOT / 'scripts/cookbook.py')} --config {unit_quote(config_path)} serve
ExecStop={unit_quote(PODMAN)} stop --ignore --time 60 {c['container_name']}
Restart=always
RestartSec=30
TimeoutStopSec=90
KillMode=process

[Install]
WantedBy=default.target
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'config.json')
    p.add_argument('command', choices=['pull', 'download', 'verify-model', 'check-image',
                   'build-kernel', 'check-processor', 'command', 'start', 'serve', 'stop', 'wait', 'status', 'install-service'])
    p.add_argument('--timeout', type=int, default=900)
    a = p.parse_args()
    c = config(a.config)
    state(c).mkdir(parents=True, exist_ok=True)
    if a.command == 'pull':
        run([PODMAN, 'pull', IMAGE])
    elif a.command == 'download':
        Path(c['model_dir']).mkdir(parents=True, exist_ok=True)
        code = ('from huggingface_hub import snapshot_download; '
                f'snapshot_download(repo_id={MANIFEST["model_repository"]!r}, '
                f'revision={MANIFEST["model_revision"]!r}, local_dir="/model")')
        run(common(c) + ['--volume', f'{c["model_dir"]}:/model', IMAGE, '-c', code])
    elif a.command == 'verify-model':
        run([sys.executable, ROOT / 'scripts/verify_model.py', c['model_dir']])
    elif a.command == 'check-image':
        check_sources()
        run(common(c) + [IMAGE, '/cookbook/scripts/check_image.py'])
    elif a.command == 'check-processor':
        run(common(c) + ['--env', 'OMP_NUM_THREADS=1', '--env', 'OPENBLAS_NUM_THREADS=1',
                        '--env', 'TOKENIZERS_PARALLELISM=false',
                        '--volume', f'{c["model_dir"]}:/model:ro', IMAGE,
                        '/cookbook/scripts/check_multimodal_processor.py'])
    elif a.command == 'build-kernel':
        (state(c) / 'extensions').mkdir(exist_ok=True)
        (state(c) / 'cache').mkdir(exist_ok=True)
        run(common(c) + ['--env', 'TORCH_CUDA_ARCH_LIST=8.0', '--env', 'MAX_JOBS=2',
                        '--volume', f'{state(c) / "cache"}:/root/.cache', IMAGE,
                        '/cookbook/scripts/build_kernel.py'])
    elif a.command == 'command':
        print(shlex.join(create_command(c)))
    elif a.command in ('start', 'serve'):
        start(c, foreground=a.command == 'serve')
    elif a.command == 'stop':
        stop(c)
    elif a.command == 'wait':
        wait(c, a.timeout)
    elif a.command == 'status':
        item = inspect(c)
        print(json.dumps(None if item is None else {'state': item['State'], 'image': item['ImageName']}, indent=2))
    elif a.command == 'install-service':
        preflight(c)
        # Check container ownership now, before enabling a boot-time service.
        owned(c)
        dest = Path.home() / '.config/systemd/user' / SERVICE
        if dest.exists() and 'scripts/cookbook.py' not in dest.read_text():
            raise ValueError(f'Refusing to overwrite an unrelated unit: {dest}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(service_text(c, a.config.resolve()))
        run(['systemctl', '--user', 'daemon-reload'])
        run(['systemctl', '--user', 'enable', SERVICE])
        print(f'Installed {dest}. Start with: systemctl --user start {SERVICE}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
