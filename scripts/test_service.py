#!/usr/bin/env python3
"""Smoke-test service restart/recovery using a disposable CPU-only container."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from cookbook import IMAGE, LABEL, service_text, unit_quote, PODMAN

podman = shutil.which('podman')
if not podman:
    raise SystemExit('Podman is required')
name = f'qwen38-cookbook-smoke-{os.getpid()}'
unit_name = name + '.service'
unit = Path.home() / '.config/systemd/user' / unit_name
unit.parent.mkdir(parents=True, exist_ok=True)
if unit.exists():
    raise SystemExit('Smoke-test unit already exists')
result = {'scope': 'Generated service/launcher with a disposable CPU container. '
          'Only Podman create is translated to a sleep process; no model or GPU is loaded.', 'checks': {}}


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True)


def inspect():
    p = subprocess.run([podman, 'inspect', name], capture_output=True, text=True)
    return json.loads(p.stdout)[0] if p.returncode == 0 else None


def wait_new(old=None, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        c = inspect()
        if c and c['State']['Running'] and (old is None or c['Id'] != old):
            return c['Id']
        time.sleep(1)
    raise RuntimeError('Service did not start/recover; inspect the user journal')


with tempfile.TemporaryDirectory(prefix='qwen38-service-smoke-') as directory:
    tmp = Path(directory)
    model = tmp / 'model'
    model.mkdir()
    for f in ['config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json', 'SHA256SUMS']:
        (model / f).touch()  # preflight fixture only; never opened by a model
    c = dict(model_dir=str(model), state_dir=str(tmp / 'state'), container_name=name,
             port=19091, tuned_all_reduce=False)
    cfg = tmp / 'config.json'
    cfg.write_text(json.dumps(c))
    bindir = tmp / 'bin'
    bindir.mkdir()
    wrapper = bindir / 'podman'
    wrapper.write_text('#!/usr/bin/env python3\nimport os,sys\nargs=sys.argv[1:]\n'
        "if args and args[0]=='create':\n"
        f" assert args[args.index('--name')+1]=={name!r}\n"
        f" args=['create','--name',{name!r},'--label',{LABEL + '=true'!r},'--entrypoint','python3',"
        f"{IMAGE!r},'-c','import signal,sys,time; signal.signal(signal.SIGTERM,lambda *_:sys.exit(0)); time.sleep(3600)']\n"
        f"os.execv({podman!r},['podman']+args)\n")
    wrapper.chmod(0o755)
    unit.write_text(service_text(c, cfg).replace(
        unit_quote('QWEN_COOKBOOK_PODMAN=' + PODMAN, command=False),
        unit_quote('QWEN_COOKBOOK_PODMAN=' + str(wrapper), command=False)))
    try:
        run(['systemd-analyze', '--user', 'verify', str(unit)])
        result['checks']['unit_valid'] = True
        run(['systemctl', '--user', 'daemon-reload'])
        run(['systemctl', '--user', 'start', unit_name])
        first = wait_new()
        result['checks']['start'] = True
        run(['systemctl', '--user', 'restart', unit_name])
        second = wait_new(first)
        result['checks']['restart_recreates_container'] = True
        run([podman, 'kill', name])
        wait_new(second)
        result['checks']['recovery_after_kill'] = True
        run(['systemctl', '--user', 'stop', unit_name])
        if inspect()['State']['Running']:
            raise RuntimeError('Container still running after intentional stop')
        result['checks']['intentional_stop'] = True
        run(['flock', '-n', str(tmp / 'state/run.lock'), 'true'])
        result['checks']['lock_released'] = True
    finally:
        subprocess.run(['systemctl', '--user', 'stop', unit_name], capture_output=True)
        subprocess.run([podman, 'rm', '--force', name], capture_output=True)
        unit.unlink(missing_ok=True)
        subprocess.run(['systemctl', '--user', 'daemon-reload'], capture_output=True)
print(json.dumps(result, indent=2))
