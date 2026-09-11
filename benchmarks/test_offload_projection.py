"""Run inside the pinned image; tests CPU-only cache-group projection."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from dataclasses import replace
import torch

spec = importlib.util.spec_from_file_location('patched_offloading', Path(__file__).resolve().parents[1] / 'overlays/offloading_connector.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheGroupSpec, FullAttentionSpec,
    CircularBufferSpec, MambaSpec)
from vllm.v1.core.sched.output import (SchedulerOutput, NewRequestData, CachedRequestData,
                                     KVConnectorBlockState)


class ProjectionTests(unittest.TestCase):
    def config(self):
        attn = FullAttentionSpec(block_size=800, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
        ring = CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
        mamba = MambaSpec(block_size=800, shapes=((4, 8),), dtypes=(torch.bfloat16,), mamba_cache_mode='all')
        return KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[name], kv_cache_spec=kv)
            for name, kv in [('attention', attn), ('scratch', ring), ('mamba', mamba)]])

    def test_scratch_is_excluded_but_physical_ids_and_input_are_preserved(self):
        config = self.config()
        p = module._OffloadGroupProjection(config)
        self.assertEqual(p.indices, (0, 2))
        self.assertEqual(p.select(([11, 12], [80], [31, 32])), ([11, 12], [31, 32]))
        self.assertEqual(len(config.kv_cache_groups), 3)
        self.assertEqual(p.config.num_blocks, 128)
        self.assertEqual([g.layer_names for g in p.config.kv_cache_groups], [['attention'], ['mamba']])
        cfg = SimpleNamespace(cache_config=SimpleNamespace(block_size=800, enable_prefix_caching=True),
                              parallel_config=SimpleNamespace(decode_context_parallel_size=1), kv_transfer_config=object())
        self.assertEqual(module.resolve_kv_cache_block_sizes(config, cfg), (800, 800))
        self.assertEqual(module.resolve_kv_cache_block_sizes(p.config, cfg), (800, 800))

    def test_new_resumed_and_boundary_handoffs_use_same_mapping(self):
        p = module._OffloadGroupProjection(self.config())
        groups = ([10], [20], [30])
        new = NewRequestData('r', [1], [], None, None, groups, 0, None)
        cached = CachedRequestData.make_empty()
        cached.new_block_ids = [groups, None]
        cached.resumed_req_ids = {'r'}
        state = KVConnectorBlockState({'r'}, lambda r: groups, {'r': [(2, 30, 800), (1, 20, 800)]})
        out = SchedulerOutput([new], cached, {'r': 1}, 1, {}, {}, [0, 0, 0], set(), [],
                              kv_connector_block_state=state)
        got = p.scheduler_output(out)
        expected = ([10], [30])
        self.assertEqual(got.scheduled_new_reqs[0].block_ids, expected)
        self.assertEqual(got.scheduled_cached_reqs.new_block_ids, [expected, None])
        self.assertEqual(got.scheduled_cached_reqs.resumed_req_ids, {'r'})
        self.assertEqual(got.kv_connector_block_state.get_block_ids('r'), expected)
        self.assertIsNone(got.kv_connector_block_state.get_block_ids('missing'))
        self.assertEqual(got.kv_connector_block_state.boundary_state_offloads, {'r': [(1, 30, 800)]})
        self.assertEqual(out.scheduled_new_reqs[0].block_ids, groups)
        self.assertEqual(out.kv_connector_block_state.boundary_state_offloads['r'][0][0], 2)

    def test_no_scratch_is_identity(self):
        config = self.config()
        config.kv_cache_groups = [config.kv_cache_groups[0], config.kv_cache_groups[2]]
        p = module._OffloadGroupProjection(config)
        self.assertIs(p.config, config)
        groups = ([1], [2])
        self.assertIs(p.select(groups), groups)
        token = object()
        self.assertIs(p.scheduler_output(token), token)

    def test_unknown_noncacheable_group_fails_closed(self):
        config = self.config()
        config.kv_cache_groups[1].kv_cache_spec = SimpleNamespace(prefix_cacheable=False)
        with self.assertRaises(NotImplementedError):
            module._OffloadGroupProjection(config)


if __name__ == '__main__':
    unittest.main()
