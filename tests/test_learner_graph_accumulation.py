"""A reused graph buffer must not overwrite previous subbatch gradients."""
from types import SimpleNamespace
import random

import pytest
import torch

from junqi.training.trainer import SelfPlayTrainer


class History:
    def __init__(self, index, length=1001):
        self.cache_key = (index, 0, length)
    def __len__(self):
        return self.cache_key[-1]


@pytest.mark.parametrize('critic', [False, True])
@pytest.mark.parametrize('fail_second', [False, True])
def test_weighted_gradients_survive_reused_storage_and_fallback(critic, fail_second):
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.device = SimpleNamespace(type='cuda')
    trainer.settings = SimpleNamespace(ppo_learner_cuda_graphs=True)
    trainer.distributed = SimpleNamespace(enabled=False)
    model = torch.nn.Linear(2, 1)
    trainer.policy = trainer.critic = model
    samples = [SimpleNamespace(state=SimpleNamespace(mode='four_dark', records=History(i)),
        value_state=None, x=torch.tensor([float(i+1), float(i*i)/11])) for i in range(12)]
    loss = model(torch.stack([s.x for s in samples])).square().mean()
    loss.backward()
    expected = [p.grad.clone() for p in model.parameters()]
    expected_loss = loss.item()
    model.zero_grad(set_to_none=True)
    shared = [torch.empty_like(p) for p in model.parameters()]
    shared_metric = torch.empty(())
    seen = []
    def graph(rows, *, critic):
        if len(rows) > 4:
            return None
        seen.append(len(rows))
        if fail_second and len(seen) == 2:
            return None
        model.zero_grad(set_to_none=True)
        part_loss = model(torch.stack([s.x for s in rows])).square().mean()
        part_loss.backward()
        with torch.no_grad():
            shared_metric.copy_(part_loss)
            for p, buffer in zip(model.parameters(), shared, strict=True):
                buffer.copy_(p.grad)
                p.grad = buffer
        return {'loss': shared_metric, **({} if critic else {'_graph_tokens': 8192.})}
    trainer._try_learner_graph = graph
    actual = trainer._backward_learner_graphs(samples, critic=critic)
    assert seen == ([4, 4] if fail_second else [4, 4, 4])
    if fail_second:
        assert actual is None and all(p.grad is None for p in model.parameters())
    else:
        assert actual['loss'].item() == pytest.approx(expected_loss)
        for p, reference in zip(model.parameters(), expected, strict=True):
            torch.testing.assert_close(p.grad, reference)


def test_full_history_value_batches_preserve_all_labels_and_reduce_repeated_prefixes():
    from junqi.training.encoding import history_prefix_groups
    t = SelfPlayTrainer.__new__(SelfPlayTrainer)
    t.settings = SimpleNamespace(ppo_minibatch_samples=512, ppo_max_samples_per_sequence=256)
    t._sequence_training_enabled = lambda: True
    samples = [SimpleNamespace(state=SimpleNamespace(mode='four_dark',
        records=History((game,player),length)), target=(game,player,length))
        for game in range(4) for length in range(1,601) for player in range(4)]
    random.seed(1257)
    old = list(t._ppo_optimizer_batches(samples,minibatch_samples=2048))
    random.seed(1257)
    new = list(t._ppo_optimizer_batches(samples,minibatch_samples=2048,full_histories=True))
    assert [len(batch) for batch in new] == [len(batch) for batch in old]
    assert sorted(id(s) for batch in new for s in batch) == sorted(map(id,samples))
    def tokens(batches):
        return sum(len(batch[group[0]].state.records) for batch in batches
            for group in history_prefix_groups([s.state for s in batch]))
    assert tokens(new) < tokens(old)


@pytest.mark.parametrize('length,prefer_graphs', [(1001, True), (700, False)])
def test_critic_padding_routing_distinguishes_full_and_partly_empty_graphs(length, prefer_graphs):
    samples = [SimpleNamespace(state=SimpleNamespace(mode='four_dark', records=History(i, length)))
               for i in range(12)]
    padding, parts = SelfPlayTrainer._critic_graph_padding_fraction(samples)
    assert parts == 3
    assert (padding <= .125) == prefer_graphs
    assert len(samples) == 12 and all(len(sample.state.records) == length for sample in samples)


@pytest.mark.parametrize('histories,uses_graph', [(12, True), (40, False)])
def test_long_critic_routing_preserves_weighted_gradients(monkeypatch, histories, uses_graph):
    from contextlib import nullcontext
    import junqi.training.trainer as module

    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.device = SimpleNamespace(type='cuda')
    trainer.settings = SimpleNamespace(ppo_learner_cuda_graphs=True,
                                      value_clip_epsilon=.2, value_coefficient=.5)
    trainer.distributed = SimpleNamespace(enabled=False)
    model = torch.nn.Linear(2, 1)
    trainer.critic = trainer.critic_parallel = model
    trainer.critic_optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    trainer.critic_grad_scaler = SimpleNamespace(scale=lambda loss: loss)
    trainer._sequence_training_enabled = lambda: True
    trainer._autocast = nullcontext
    trainer._learner_chunks = lambda rows: [rows[:7], rows[7:]]
    samples = [SimpleNamespace(state=SimpleNamespace(mode='four_dark', records=History(i)),
        value_state=None, x=torch.tensor([float(i)/histories, float(i % 5)])) for i in range(histories)]
    def loss(rows):
        return model(torch.stack([row.x for row in rows])).square().mean()
    expected_loss = loss(samples)
    expected_loss.backward()
    expected = [p.grad.clone() for p in model.parameters()]
    calls = []
    def graph(rows, *, critic):
        calls.append(len(rows))
        value = loss(rows)
        value.backward()
        return {'loss/critic_total': value.detach()}
    trainer._backward_learner_graphs = graph
    def packed(_model, rows, **kwargs):
        value = loss(rows)
        return SimpleNamespace(loss=value, metrics={'loss/critic_total': value.detach()})
    monkeypatch.setattr(module, 'critic_ppo_loss', packed)
    metrics = trainer._backward_critic_epoch(samples, defer_metrics=True)
    assert bool(calls) is uses_graph
    assert float(metrics['loss/critic_total']) == pytest.approx(float(expected_loss.detach()))
    for parameter, reference in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, reference)
