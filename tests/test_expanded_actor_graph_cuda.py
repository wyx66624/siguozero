"""Expanded graph executes the same complete histories and PPO gradients."""
import copy
from dataclasses import replace

import pytest
import torch

from junqi.training.history_arrays import ArrayHistory
from junqi.training.learner_graph import PackedLearnerGraph
from junqi.training.models import GamePolicyTransformer, ModelConfig
from junqi.training.ppo import PPOSample, policy_ppo_loss
from test_ppo import states


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph equivalence')
def test_expanded_actor_graph_matches_packed_forward_and_backward():
    torch.set_num_threads(1)
    torch.manual_seed(516)
    config = replace(ModelConfig.tiny(), max_transitions=300,
        ppo_tensor_learner=True, ppo_varlen_attention=True, temporal_causal_sdpa=True)
    model = GamePolicyTransformer(config).cuda()
    reference = copy.deepcopy(model)
    state = states(1)[0]
    record = state.records[0]
    samples = []
    for i in range(36):
        history = ArrayHistory(state.mode, [record] * (150 + i % 20), 300, (i, 0))
        view = replace(state, records=history.view())
        samples.append(PPOSample(view, view.legal_actions[i % len(view.legal_actions)],
            -4., 0., (-1.) ** i * .5, .1, 0))
    graph = PackedLearnerGraph(model, critic=False, amp_dtype=torch.bfloat16,
        tokens=8192, sequences=64, sources=2048)
    for advantage_shift in (0., .2):
        batch = [replace(s, advantage=s.advantage + advantage_shift) for s in samples]
        reference.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = policy_ppo_loss(reference, batch, clip_epsilon=.2,
                entropy_coefficient=.01, sequence_training=True, defer_metrics=True)
        expected.loss.backward()
        expected_loss = expected.loss.detach().clone()
        del expected
        assert graph.stage(model, batch, clip=.2, coefficient=.01,
                           opening_coefficient=None, opening_plies=16)
        actual = graph.run(model)
        torch.testing.assert_close(actual['loss/policy_total'], expected_loss, rtol=.03, atol=.003)
        for parameter, wanted in zip(model.parameters(), reference.parameters(), strict=True):
            assert (parameter.grad is None) == (wanted.grad is None)
            if wanted.grad is not None:
                assert torch.isfinite(parameter.grad).all()
                torch.testing.assert_close(parameter.grad, wanted.grad, rtol=.04, atol=.003)
    assert graph.captures == 1 and graph.replays == 2
