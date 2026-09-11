import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cookbook', ROOT / 'scripts/cookbook.py')
cookbook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cookbook)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='qwen test ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.c = dict(model_dir=str(self.root / 'model files'), state_dir=str(self.root / 'state files'),
                      container_name='cookbook-test', port=19088, tuned_all_reduce=False)
        for name in ['config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json', 'SHA256SUMS']:
            p = Path(self.c['model_dir']) / name
            p.parent.mkdir(exist_ok=True)
            p.touch()

    def test_command_preserves_spaces_and_precision(self):
        args = cookbook.create_command(self.c)
        self.assertIn(self.c['model_dir'] + ':/model:ro', args)
        self.assertIn(cookbook.IMAGE, args)
        self.assertEqual(args[args.index('--max-model-len') + 1], '262144')
        self.assertNotIn('--kv-cache-dtype', args)
        self.assertFalse(any('NCCL_ALGO=' in x for x in args))
        self.assertFalse(any('/opt/q38-ar' in x for x in args))
        self.c['tuned_all_reduce'] = True
        self.assertIn('QWEN_SM80_AR_TUNING=1', cookbook.create_command(self.c))

    def test_refuses_foreign_container_before_stop(self):
        with patch.object(cookbook, 'unit_active', return_value=False), \
             patch.object(cookbook, 'inspect', return_value={'Config': {'Labels': {}}}), \
             patch.object(cookbook, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'another deployment'):
                cookbook.start(self.c)
            run.assert_not_called()

    def test_start_does_not_pass_lock_to_children(self):
        commands = []
        def child(args, **kwargs):
            commands.append(args)
            # Same-process /proc scan detects the actual descriptor; an exec'd
            # child must not inherit it (the old launcher failed this contract).
            lock_path = str(Path(self.c['state_dir']) / 'run.lock')
            p = subprocess.run(['python3', '-c',
                'import os,sys; print(any(os.path.realpath("/proc/self/fd/"+f)==sys.argv[1] for f in os.listdir("/proc/self/fd")))', lock_path],
                close_fds=False, capture_output=True, text=True, check=True)
            self.assertEqual(p.stdout.strip(), 'False')
        with patch.object(cookbook, 'unit_active', return_value=False), \
             patch.object(cookbook, 'inspect', return_value=None), patch.object(cookbook, 'run', side_effect=child):
            cookbook.start(self.c)
        self.assertEqual([x[1] for x in commands], ['create', 'start'])
        p = subprocess.run(['flock', '-n', str(Path(self.c['state_dir']) / 'run.lock'), 'true'])
        self.assertEqual(p.returncode, 0)

    def test_failed_stop_does_not_remove_container(self):
        item = {'Config': {'Labels': {cookbook.LABEL: 'true'}}, 'State': {'Running': True}}
        with patch.object(cookbook, 'unit_active', return_value=False), \
             patch.object(cookbook, 'inspect', return_value=item), \
             patch.object(cookbook, 'run', side_effect=subprocess.CalledProcessError(1, 'stop')) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                cookbook.start(self.c)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][1], 'stop')

    def test_systemd_is_the_only_manager_when_active(self):
        with patch.object(cookbook, 'unit_active', return_value=True):
            with self.assertRaisesRegex(ValueError, 'systemctl'):
                cookbook.start(self.c)
            with self.assertRaisesRegex(ValueError, 'systemctl'):
                cookbook.stop(self.c)

    def test_service_uses_foreground_and_restart(self):
        text = cookbook.service_text(self.c, self.root / 'config with spaces.json')
        self.assertIn('Restart=always', text)
        self.assertIn(' serve\n', text)
        self.assertIn('stop --ignore --time 60 cookbook-test', text)
        self.assertIn('KillMode=process', text)
        self.assertIn('QWEN_COOKBOOK_PODMAN=' + cookbook.PODMAN, text)
        self.assertEqual(cookbook.unit_quote('/tmp/$var', command=False), '"/tmp/$var"')
        self.assertEqual(cookbook.unit_quote('/tmp/with %n/$var'), '"/tmp/with %%n/$$var"')

    def test_historical_final_command_matches_launcher(self):
        import shlex
        self.c['tuned_all_reduce'] = True
        cfg = self.root / 'config.json'
        cfg.write_text(json.dumps(self.c))
        output = subprocess.check_output(['python3', str(ROOT / 'scripts/experiment_command.py'),
                    '09-custom-ar-kernel-tuned', '--config', str(cfg)], text=True)
        self.assertEqual(shlex.split(output), cookbook.create_command(self.c))

    def test_historical_baseline_has_original_graph_and_collective_settings(self):
        import shlex
        cfg = self.root / 'config.json'
        cfg.write_text(json.dumps(self.c))
        output = subprocess.check_output(['python3', str(ROOT / 'scripts/experiment_command.py'),
                    '00-baseline', '--config', str(cfg)], text=True)
        args = shlex.split(output)
        self.assertEqual(json.loads(args[args.index('--compilation-config') + 1])['cudagraph_mode'], 'NONE')
        self.assertIn('--disable-custom-all-reduce', args)
        self.assertNotIn('--speculative-config', args)
        self.assertIn('NCCL_ALGO=Ring', args)
        self.assertFalse(any('mamba_hybrid.py:' in a for a in args))

    def test_missing_kernel_fails_before_container_mutation(self):
        self.c['tuned_all_reduce'] = True
        with self.assertRaises(FileNotFoundError):
            cookbook.preflight(self.c)


if __name__ == '__main__':
    unittest.main()
