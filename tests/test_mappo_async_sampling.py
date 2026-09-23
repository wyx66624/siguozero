"""Deferred results preserve action/RNG ordering, ownership and complete games."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from junqi.training.mappo import collect_mappo_samples
from junqi.training.models import (GamePolicyTransformer, GameCentralValueTransformer,
                                   PieceConditionedLayoutPointerDecoder)
from junqi.training.ppo import FrozenValueActor
from junqi.training.ppo_environment import ParallelPPOEnvironment
from junqi.training.ppo_sampling import PendingActionSamples
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor
from test_mappo import MODE, histories, tiny_config


@pytest.mark.parametrize('groups', [1, 2])
def test_pipeline_result_wait_follows_environment_receive_and_keeps_rng(groups):
    torch.set_num_threads(1)
    config = tiny_config(ppo_fixed_kv=False)
    results = []
    for deferred in (False, True):
        torch.manual_seed(192)
        policy = GamePolicyTransformer(config).eval()
        critic = GameCentralValueTransformer(config).eval()
        layout = PieceConditionedLayoutPointerDecoder(config).eval()
        pool = BaseGamePool(MODE, pool_size=3, max_transitions=16, max_game_plies=4,
                            seed=43, centralized_critic=True)
        actor = FrozenPolicyActor(policy, max_batch_size=3)
        environment = ParallelPPOEnvironment(2)
        received = 0
        resolved = 0
        original_receive = environment.receive
        def receive():
            nonlocal received
            value = original_receive()
            received += 1
            return value
        def sample_async(states, **options):
            nonlocal resolved
            value = actor.sample(states, **options)
            submitted = received
            def finish():
                nonlocal resolved
                assert received > submitted, 'result was read before receiving the previous wave'
                resolved += 1
                return value
            return SimpleNamespace(result=finish) if deferred else value
        try:
            with patch.object(actor, 'sample_async', side_effect=sample_async), \
                    patch.object(environment, 'receive', side_effect=receive):
                samples, outcomes, metrics = collect_mappo_samples(
                    pool, actor, FrozenValueActor(critic, deferred=True, amp_dtype=None,
                                                 max_batch_size=3), layout,
                    count=23, behavior_version=7, discount=1., gae_lambda=.95,
                    environment=environment, pipeline_groups=groups,
                    advantage_mode='player_decisions', reward_scope='player')
            assert metrics.environment_plies == len(samples) == 23
            assert metrics.base_games_completed > 0 and len(outcomes) > 0
            assert metrics.actor_deferred_batches == resolved
            assert bool(resolved) == (deferred and groups > 1)
            signature = [(s.action, s.old_log_prob, s.advantage, s.value_target, s.value_player,
                          tuple(s.state.records), tuple(s.value_state.records),
                          tuple((v.player, v.old_value, v.target, tuple(v.state.records))
                                for v in s.additional_values)) for s in samples]
            results.append((signature, pool.state_dict(), torch.get_rng_state()))
        finally:
            environment.close()
    assert results[0][:2] == results[1][:2]
    assert torch.equal(results[0][2], results[1][2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA result ownership')
def test_deferred_host_result_survives_source_overwrite_and_is_repeatable():
    data = torch.tensor([[3., 9., -.25], [2., 4., -.5]], device='cuda')
    pending = PendingActionSamples(data, return_log_probs=True)
    data.fill_(-999)  # Another graph replay may overwrite the same output.
    actions, logs = pending.result()
    assert actions == [[(3, 9)], [(2, 4)]]
    torch.testing.assert_close(torch.cat(logs), torch.tensor([-.25, -.5]))
    assert pending.result() is pending.result()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph sampling')
@pytest.mark.parametrize('return_log_probs', [False, True])
def test_cuda_actor_deferred_sampling_preserves_actions_logs_and_rng(return_log_probs):
    torch.set_num_threads(1)
    torch.manual_seed(812)
    config = replace(tiny_config(), ppo_cuda_graphs=True, ppo_sampling_graphs=True,
                     ppo_fixed_kv=True, temporal_causal_sdpa=True)
    model = GamePolicyTransformer(config).cuda().eval()
    game, history, _ = histories()
    states = [history.state_for(game, player=p) for p in range(4)]
    result = []
    for deferred in (False, True):
        model.clear_inference_board_cache()
        actor = FrozenPolicyActor(model, amp_dtype=torch.bfloat16, max_batch_size=8)
        model.start_ppo_inference_cache(capacity=4, behavior_version=0)
        assert actor.sample_async([], return_log_probs=return_log_probs) == ([], [])
        torch.cuda.manual_seed(212)
        output = (actor.sample_async(states, return_log_probs=return_log_probs) if deferred
                  else actor.sample(states, return_log_probs=return_log_probs))
        if deferred:
            assert isinstance(output, PendingActionSamples)
            # Layout work may be queued on the same stream before action use.
            torch.rand(17, device='cuda')
            output = output.result()
        else:
            torch.rand(17, device='cuda')
        result.append((output, torch.cuda.get_rng_state()))
    assert result[0][0][0] == result[1][0][0]
    if return_log_probs:
        torch.testing.assert_close(torch.cat(result[0][0][1]), torch.cat(result[1][0][1]), rtol=0, atol=0)
    assert torch.equal(result[0][1], result[1][1])
    model.clear_inference_board_cache()
