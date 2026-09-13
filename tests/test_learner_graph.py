"""CPU loss equivalence for bounded graph padding, heads and phase masks."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from junqi.training.history_arrays import ArrayHistory
from junqi.training.learner_graph import PackedLearnerGraph
from junqi.training.models import GamePolicyTransformer, GameValueTransformer, ModelConfig, PolicyFeatures
from junqi.training.ppo import PPOSample, policy_ppo_loss, critic_ppo_loss
from junqi.training.accelerator import trim_cuda_cache
from types import SimpleNamespace
from test_ppo import states


class LearnerGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def graph(self, model, critic):
        graph = PackedLearnerGraph(model, critic=critic, amp_dtype=None, tokens=64)
        graph.SEQUENCES, graph.SAMPLES, graph.ACTIONS, graph.SOURCES = 8, 8, 2048, 256
        def buffer(name, shape, dtype):
            if name not in graph.host:
                graph.host[name] = torch.empty(shape, dtype=dtype)
                graph.device[name] = torch.empty(shape, dtype=dtype)
            return graph.host[name].numpy()
        graph._buffer = buffer
        return graph

    def batch(self):
        batch = states(3)
        batch[0] = replace(batch[0], legal_actions=(batch[0].legal_actions[0],))
        batch = [replace(state, records=ArrayHistory(state.mode, state.records, 16, (i,0)).view())
                 for i,state in enumerate(batch)]
        return [PPOSample(state,state.legal_actions[-1],-.8-i,.15*i,(-1.)**i,.4-i*.3,0,
                          old_draw_value=-.04,draw_value_target=-.11) for i,state in enumerate(batch)]

    def test_padding_preserves_policy_critic_losses_and_gradients(self):
        for critic in (False, True):
            torch.manual_seed(83)
            model = (GameValueTransformer if critic else GamePolicyTransformer)(ModelConfig.tiny())
            reference = copy.deepcopy(model)
            samples = self.batch()
            contexts = torch.randn(8, model.config.temporal_dim)
            reference_features = PolicyFeatures(contexts[:3],torch.ones(3,129,dtype=torch.bool))
            graph = self.graph(model,critic)
            graph.callable = lambda *args: contexts
            self.assertTrue(graph.stage(model,samples,clip=.2,coefficient=.5 if critic else .01,
                                        opening_coefficient=None if critic else .04,opening_plies=1))
            loss,metrics = graph._loss(model)
            with patch.object(reference,'_encode_full',return_value=reference_features):
                output = (critic_ppo_loss(reference,samples,clip_epsilon=.2,value_coefficient=.5,sequence_training=True,defer_metrics=True)
                          if critic else policy_ppo_loss(reference,samples,clip_epsilon=.2,entropy_coefficient=.01,
                              opening_entropy_coefficient=.04,entropy_opening_plies=1,sequence_training=True,defer_metrics=True))
            torch.testing.assert_close(loss,output.loss,rtol=2e-6,atol=2e-7)
            expected = torch.stack(list(output.metrics.values()))
            torch.testing.assert_close(metrics,expected,rtol=2e-6,atol=2e-7)
            loss.backward();output.loss.backward()
            for actual,wanted in zip(model.parameters(),reference.parameters(),strict=True):
                self.assertEqual(actual.grad is None,wanted.grad is None)
                if wanted.grad is not None:
                    torch.testing.assert_close(actual.grad,wanted.grad,rtol=2e-5,atol=3e-7)

    def test_capacity_rejects_without_truncating_or_allocating(self):
        model=GamePolicyTransformer(ModelConfig.tiny())
        graph=self.graph(model,False)
        samples=self.batch()
        options=dict(clip=.2,coefficient=.01,opening_coefficient=.04,opening_plies=1)
        self.assertFalse(graph.stage(model,samples*3,**options))
        self.assertFalse(graph.host)
        graph.TOKENS=2
        self.assertFalse(graph.stage(model,samples,**options))
        self.assertFalse(graph.host)
        self.assertEqual(len(samples),3)

    def test_sparse_sampling_masks_equal_dense_reduction(self):
        model=GamePolicyTransformer(ModelConfig.tiny())
        batch=[item.state for item in self.batch()]
        source,destination=model._sampling_legal_masks(batch,129)
        expected=torch.zeros((3,129,129),dtype=torch.bool)
        for row,state in enumerate(batch):
            for left,right in state.legal_actions:
                expected[row,left,right]=True
        torch.testing.assert_close(destination,expected,rtol=0,atol=0)
        torch.testing.assert_close(source,expected.any(dim=-1),rtol=0,atol=0)

    def test_cache_trim_uses_only_unused_blocks_at_high_watermark(self):
        gib=1024**3
        with patch('torch.cuda.memory_reserved',side_effect=[22*gib,19*gib]), \
                patch('torch.cuda.memory_allocated',return_value=18*gib), \
                patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=24*gib)), \
                patch('torch.cuda.empty_cache') as release:
            self.assertEqual(trim_cuda_cache('cuda'),3*gib)
            release.assert_called_once_with()
        for reserved,allocated in ((19*gib,15*gib),(22*gib,int(21.5*gib))):
            with patch('torch.cuda.memory_reserved',return_value=reserved), \
                    patch('torch.cuda.memory_allocated',return_value=allocated), \
                    patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=24*gib)), \
                    patch('torch.cuda.empty_cache') as release:
                self.assertEqual(trim_cuda_cache('cuda'),0)
                release.assert_not_called()
        self.assertEqual(trim_cuda_cache('cpu'),0)

    def test_bounded_prefill_restores_input_order_and_exact_contexts(self):
        model = GamePolicyTransformer(ModelConfig.tiny()).eval()
        reference = copy.deepcopy(model)
        batch = [sample.state for sample in reversed(self.batch())]
        model.start_ppo_inference_cache(capacity=len(batch), behavior_version=1)
        store = model._fixed_kv_store
        store.PREFILL_PADDED_TOKENS = max(len(state.records) for state in batch)
        chunks = list(store._prefill_batches(batch, range(len(batch))))
        self.assertGreater(len(chunks), 1)
        self.assertEqual(sorted(i for chunk in chunks for i in chunk), list(range(len(batch))))
        for chunk in chunks:
            self.assertLessEqual(len(chunk) * max(len(batch[i].records) for i in chunk),
                                 store.PREFILL_PADDED_TOKENS)
        with torch.inference_mode():
            actual = model.encode(batch).context
            expected = reference.encode(batch).context
            torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
            torch.testing.assert_close(model.encode(batch).context, actual, rtol=0, atol=0)
        self.assertEqual(store.prefill_states, len(batch))


if __name__=='__main__':
    unittest.main()
