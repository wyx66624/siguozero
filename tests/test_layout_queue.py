"""Layout freshness, exact teacher forcing and optimizer-batch normalization."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from junqi.training.layout_buffer import LayoutOutcomeBuffer
from junqi.training.losses import layout_grpo_loss, prepare_layout_batch
from junqi.training.models import ModelConfig, PieceConditionedLayoutPointerDecoder, PreNormEncoderBlock
from junqi.training.modes import TrainingMode
from junqi.training.rollout import LayoutOutcome


def sequential_replay(model, choices, modes):
    occupants = torch.zeros_like(choices)
    rows = torch.arange(len(choices))
    logs, entropies = [], []
    for step in range(25):
        steps = torch.full((len(choices),), step)
        lp = F.log_softmax(model.logits(occupants, steps, modes) / .7, dim=-1)
        logs.append(lp[rows, choices[:, step]])
        entropies.append(-(lp.exp() * lp.nan_to_num()).sum(-1))
        occupants = occupants.clone()
        occupants[rows, choices[:, step]] = model.piece_sequence[step] + 1
    return torch.stack(logs, 1), torch.stack(entropies, 1)


class LayoutQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(621)
        cls.model = PieceConditionedLayoutPointerDecoder(ModelConfig.tiny()).eval()
        cls.samples = cls.model.sample_layouts(7, TrainingMode.FOUR_DARK)

    def outcome(self, version, reward=1., index=0):
        return LayoutOutcome(self.samples[index], reward, index % 4, version)

    def test_ring_wrap_expiry_and_exact_accounting(self):
        buffer = LayoutOutcomeBuffer(4, 2)
        buffer.advance(10)
        buffer.extend([self.outcome(7), self.outcome(8, -1), self.outcome(9, -.15),
                       self.outcome(10), self.outcome(10, -1), self.outcome(10, -.15)])
        self.assertEqual(len(buffer), 4)
        self.assertEqual(buffer.totals, dict(enqueued=6, consumed=0, expired=1, overflow=1))
        self.assertEqual([x.behavior_version for x in buffer.take(2)], [9, 10])
        buffer.extend([self.outcome(11), self.outcome(11, -1)])
        buffer.advance(13)
        self.assertEqual([x.behavior_version for x in buffer], [11, 11])
        self.assertEqual(buffer.totals['enqueued'], len(buffer) + sum(buffer.totals[k] for k in ('consumed','expired','overflow')))
        self.assertEqual(buffer[0].sample.setup, self.samples[0].setup)

    def test_legacy_restore_filters_before_reconstructing_and_preserves_stats(self):
        source = LayoutOutcomeBuffer(20, 0)
        source.extend(self.outcome(v, -.15 if v % 2 else 1) for v in range(20))
        buffer = LayoutOutcomeBuffer(4, 3)
        buffer.restore(source.records(), version=19, minimum_version=18)
        self.assertEqual([x.behavior_version for x in buffer], [18, 19])
        self.assertEqual(buffer.totals['expired'], 18)
        metrics = buffer.metrics()
        self.assertEqual(metrics['layout/enqueued'], 0)
        self.assertEqual(metrics['layout/expired'], 18)
        buffer.take(1)
        restored = LayoutOutcomeBuffer(4, 3)
        restored.restore(buffer.records(), version=19, totals=buffer.totals)
        self.assertEqual(restored.records(), buffer.records())
        self.assertEqual(restored.totals, buffer.totals)

    def test_production_consumes_variable_batches_without_growth(self):
        buffer = LayoutOutcomeBuffer(8192, 8)
        for version in range(20):
            buffer.advance(version)
            buffer.extend(self.outcome(version, (-1., -.15, 1.)[i % 3]) for i in range(944))
            outcomes = buffer.take(2048)
            self.assertEqual(len(outcomes), 944)
            self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.totals['expired'], 0)
        self.assertEqual(buffer.totals['overflow'], 0)

    def test_batched_prefixes_match_sequential_values_and_gradients(self):
        a, b = copy.deepcopy(self.model).train(), copy.deepcopy(self.model).train()
        choices = torch.tensor([s.position_indices for s in self.samples[:3]])
        modes = torch.tensor([0, 1, 2])
        actual = a.evaluate_layouts(choices, modes)
        expected = sequential_replay(b, choices, modes)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)
        (actual[0].mean() + .1 * actual[1].mean()).backward()
        (expected[0].mean() + .1 * expected[1].mean()).backward()
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=2e-4, atol=3e-6)

    def test_unmasked_attention_matches_previous_attention_and_gradients(self):
        a = PreNormEncoderBlock(32, 4, 64, 0.)
        b = copy.deepcopy(a)
        x = torch.randn(3, 26, 32, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        valid = torch.ones(3, 26, dtype=torch.bool)
        actual, expected = a(x, valid_mask=valid, unmasked=True), b(y, valid_mask=valid)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        actual.square().mean().backward(); expected.square().mean().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=2e-5, atol=2e-6)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6)

    def test_microbatches_preserve_global_advantages_and_gradient(self):
        a, b, ref = copy.deepcopy(self.model), copy.deepcopy(self.model), copy.deepcopy(self.model).eval()
        outcomes = [self.outcome(3, [-1., -.15, 1., 1., -.15, -1., 1.][i], i) for i in range(7)]
        kwargs = dict(clip_epsilon=.2, kl_coefficient=.02, entropy_coefficient=.01)
        whole = layout_grpo_loss(a, ref, outcomes, **kwargs)
        whole.loss.backward()
        batch = prepare_layout_batch(outcomes, device='cpu')
        total = 0.
        for start in range(0, 7, 3):
            stop = min(start + 3, 7)
            part = layout_grpo_loss(b, ref, (), prepared=batch.slice(start, stop), **kwargs)
            weight = (stop-start)/7
            (part.loss * weight).backward()
            total += part.loss.item() * weight
        self.assertAlmostEqual(total, whole.loss.item(), places=6)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=3e-6)

    def test_all_draw_batch_has_zero_task_advantage_but_retains_regularization(self):
        outcomes = [self.outcome(3, -.15, i) for i in range(4)]
        output = layout_grpo_loss(self.model, self.model, outcomes,
                                 clip_epsilon=.2, kl_coefficient=.02, entropy_coefficient=.01)
        self.assertEqual(output.metrics['layout/zero_advantage_batch'], 1.)
        self.assertEqual(output.metrics['loss/layout_grpo'], 0.)
        self.assertLess(output.metrics['loss/layout_total'], 0.)

    def test_replay_rejects_duplicate_and_illegal_piece_positions(self):
        choices = torch.tensor([self.samples[0].position_indices])
        choices[0, 1] = choices[0, 0]
        with self.assertRaisesRegex(ValueError, 'illegal position'):
            self.model.evaluate_layouts(choices, torch.tensor([0]))

    def test_default_age_limit_preserves_grpo_mode_isolation(self):
        from junqi.training.settings import TrainingSettings
        config = Path(__file__).parents[1] / 'configs/bootstrap.yaml'
        grpo = TrainingSettings.from_yaml(config, 'two_player', tiny=True)
        ppo = TrainingSettings.from_yaml(config, 'four_dark', tiny=True)
        self.assertEqual(grpo.layout_max_behavior_age, 0)
        self.assertEqual(ppo.layout_max_behavior_age, 16)

    def test_settings_and_exact_trainer_resume(self):
        from junqi.training.settings import TrainingSettings
        from junqi.training.trainer import SelfPlayTrainer
        settings = TrainingSettings.from_yaml(Path(__file__).parents[1]/'configs/bootstrap.yaml',
            'two_player', tiny=True, overrides=dict(device='cpu', layout_outcomes_per_update=8,
                                                   layout_buffer_capacity=16, layout_max_behavior_age=2,
                                                   layout_microbatch_size=3))
        with tempfile.TemporaryDirectory() as directory:
            trainer = SelfPlayTrainer(settings, run_directory=directory, auto_resume=False)
            trainer.update = 10
            sample = trainer.layout.sample_layouts(1, TrainingMode.TWO_PLAYER)[0]
            trainer.layout_buffer.append(LayoutOutcome(sample, 1., 0, 1))
            trainer.layout_buffer.append(LayoutOutcome(sample, -.15, 0, 10))
            trainer.save_checkpoint(reason='queue-test', archive=False)
            original = {k: v.clone() for k,v in trainer.policy.state_dict().items()}
            trainer.logger.close()
            resumed = SelfPlayTrainer(settings, run_directory=directory)
            try:
                self.assertEqual(resumed.update, 10)
                self.assertEqual(len(resumed.layout_buffer), 1)
                self.assertEqual(resumed.layout_buffer[0].behavior_version, 10)
                for key, value in resumed.policy.state_dict().items():
                    torch.testing.assert_close(value, original[key], rtol=0, atol=0)
                resumed.layout_buffer.append(LayoutOutcome(sample, 1., 0, 10))
                metrics = resumed._update_layout()
                self.assertEqual(metrics['optimizer/layout_samples'], 2)
                self.assertEqual(metrics['optimizer/layout_steps'], 1)
                self.assertEqual(metrics['layout/buffer_remaining'], 0)
            finally:
                resumed.logger.close()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA graph validation requires CUDA')
    def test_cuda_layout_graph_preserves_sampling_rng_and_copy_lifecycle(self):
        config = replace(ModelConfig.tiny(), ppo_sampling_graphs=True)
        model = PieceConditionedLayoutPointerDecoder(config).cuda().eval()
        torch.cuda.manual_seed_all(251)
        uniforms = torch.rand(4, 25, device='cuda')
        rng = torch.cuda.get_rng_state().clone()
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            expected = model._sample_layout_tensors(4, 0, .7, uniforms)
            actual = model._sample_layout_graph(4, 0, .7, uniforms)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
            # Captured weights stay live after optimizer-like in-place updates.
            with torch.no_grad():
                model.query[-1].weight.add_(.01)
            expected = model._sample_layout_tensors(4, 0, .7, uniforms)
            actual = model._sample_layout_graph(4, 0, .7, uniforms)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            clone = copy.deepcopy(model)
            self.assertEqual(len(clone._layout_sampling_graphs), 0)
            rng = torch.cuda.get_rng_state().clone()
            first = model.sample_layouts(4, TrainingMode.FOUR_DARK)
            torch.cuda.set_rng_state(rng)
            second = clone.sample_layouts(4, TrainingMode.FOUR_DARK)
            self.assertEqual([s.position_indices for s in first], [s.position_indices for s in second])
            self.assertTrue(all(len(s.setup)==25 for s in first))
        model.cpu()
        self.assertEqual(len(model._layout_sampling_graphs), 0)


if __name__ == '__main__':
    unittest.main()
