"""Regression: backend alignment changes the cache block size after construction."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

class CacheBlockSizeTest(unittest.TestCase):
    def state(self, resolved_spec=True, block_size=800):
        state=object.__new__(MambaHybridModelState)
        state._align_mode=True
        state._mamba_block_size=16  # Value before platform alignment.
        state._mamba_spec=SimpleNamespace(block_size=block_size) if resolved_spec else None
        state.cache_config=SimpleNamespace(mamba_block_size=block_size,block_size=block_size)
        state.num_accepted_tokens_gpu=torch.full((1,),3,dtype=torch.int32)
        state._mamba_state_idx_gpu=torch.zeros(1,dtype=torch.int32)
        return state

    def seed(self,state,computed):
        with patch.object(DefaultModelState,'add_request',return_value=None):
            state.add_request(0,SimpleNamespace(num_computed_tokens=computed))
        self.assertEqual(state.num_accepted_tokens_gpu.item(),1)
        return state._mamba_state_idx_gpu.item()

    def test_resumed_prefix_uses_finalized_blocks(self):
        for block_size,cases in [(800,[(0,-1),(800,0),(1600,1),(2400,2),(2401,3)]),(784,[(0,-1),(784,0),(1568,1),(2352,2),(2353,3)])]:
            state=self.state(block_size=block_size)
            for computed,expected in cases:
                with self.subTest(computed=computed,block_size=block_size):self.assertEqual(self.seed(state,computed),expected)

    def test_first_request_uses_live_configuration(self):
        self.assertEqual(self.seed(self.state(resolved_spec=False),2400),2)

    def test_spec_is_authoritative_over_config(self):
        state=self.state();state.cache_config.mamba_block_size=16
        self.assertEqual(self.seed(state,2400),2)

if __name__=='__main__':unittest.main()
