"""Numerical and optimizer-state contracts for coalesced learner transfers."""
import copy
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from junqi.training.entropy import policy_entropy_bonus
from junqi.training.ppo import _upload_targets, critic_ppo_loss, policy_ppo_loss, PPOSample
from junqi.training.models import GamePolicyTransformer, GameValueTransformer, ModelConfig
from junqi.training.trainer import SelfPlayTrainer
from test_ppo import states


class LearnerTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_coalesced_targets_equal_separate_float32_columns(self):
        rows = [(-4.12345678, 1.7, 0.0, -0.15), (-2., -0.7, .01, .2)]
        actual = _upload_targets(rows, torch.zeros(1))
        expected = torch.stack([torch.tensor([r[i] for r in rows]) for i in range(4)], dim=1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_entropy_metadata_preserves_loss_and_gradient(self):
        batch = states(3)
        batch[0] = replace(batch[0], legal_actions=(batch[0].legal_actions[0],))
        left = torch.tensor([0., 1.2, .8], requires_grad=True)
        right = left.detach().clone().requires_grad_()
        metadata = _upload_targets([(math.log(len(s.legal_actions)) if len(s.legal_actions)>1 else 0.,
                                     float(len(s.records)<=1)) for s in batch], left)
        old, old_metrics = policy_entropy_bonus(batch, left, coefficient=.01, opening_coefficient=.04, opening_plies=1)
        new, new_metrics = policy_entropy_bonus(batch, right, coefficient=.01, opening_coefficient=.04,
                                                 opening_plies=1, metadata=metadata)
        old.backward();new.backward()
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)
        for key in old_metrics:torch.testing.assert_close(new_metrics[key],old_metrics[key],rtol=0,atol=0)

    def test_clipping_and_adam_states_match_existing_path(self):
        torch.manual_seed(119)
        old = torch.nn.Sequential(torch.nn.Linear(5,7),torch.nn.Tanh(),torch.nn.Linear(7,2))
        new = copy.deepcopy(old)
        optimizers = [torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.05) for m in (old,new)]
        trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
        trainer.settings = SimpleNamespace(gradient_norm_clip=.1)
        for _ in range(3):
            data = torch.randn(11,5)
            losses = [m(data).square().mean() for m in (old,new)]
            for loss in losses:loss.backward()
            expected = torch.nn.utils.clip_grad_norm_(old.parameters(),.1,error_if_nonfinite=True)
            with patch.object(trainer, '_materialize_metrics', wraps=trainer._materialize_metrics) as copies:
                values,norm = trainer._clip_and_materialize(new,{'loss':losses[1].detach()})
                self.assertEqual(copies.call_count,1)
            self.assertEqual(norm,float(expected))
            self.assertEqual(values['loss'],float(losses[0].detach()))
            for a,b in zip(old.parameters(),new.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
            for optimizer in optimizers:optimizer.step();optimizer.zero_grad(set_to_none=True)
            for a,b in zip(old.parameters(),new.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            for a,b in zip(optimizers[0].state.values(),optimizers[1].state.values()):
                for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_nonfinite_gradient_aborts_before_optimizer_mutation(self):
        trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
        trainer.settings = SimpleNamespace(gradient_norm_clip=1.)
        for invalid in (float('nan'),float('inf')):
            model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters())
            before=copy.deepcopy(model.state_dict())
            for p in model.parameters():p.grad=torch.full_like(p,invalid)
            with self.assertRaisesRegex(RuntimeError,'non-finite'):
                trainer._clip_and_materialize(model,{'loss':torch.tensor(0.)})
                optimizer.step()
            self.assertEqual(len(optimizer.state),0)
            for key,value in before.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)

    def test_losses_request_one_target_upload_and_keep_frozen_policy_excluded(self):
        config=ModelConfig.tiny();policy=GamePolicyTransformer(config);critic=GameValueTransformer(config)
        batch=states(3)
        samples=[PPOSample(s,s.legal_actions[0],-4.,.1,.2,.4,0,old_draw_value=-.02,draw_value_target=-.05,
                           learnable=i!=1) for i,s in enumerate(batch)]
        with patch('junqi.training.ppo._upload_targets',wraps=_upload_targets) as uploads:
            result=policy_ppo_loss(policy,samples,clip_epsilon=.2,entropy_coefficient=.01,
                                   opening_entropy_coefficient=.04,entropy_opening_plies=16)
            self.assertEqual(uploads.call_count,1)
            self.assertEqual(len(uploads.call_args.args[0]),2)
            result.loss.backward()
        with patch('junqi.training.ppo._upload_targets',wraps=_upload_targets) as uploads:
            result=critic_ppo_loss(critic,samples,clip_epsilon=.2,value_coefficient=.5)
            self.assertEqual(uploads.call_count,1)
            self.assertEqual(len(uploads.call_args.args[0]),3)
            result.loss.backward()


if __name__=='__main__':unittest.main()
