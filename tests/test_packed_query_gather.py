"""Sparse last-layer outputs must retain dense causal outputs and gradients."""
import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from junqi.training import packed_attention
from junqi.training.models import PreNormEncoderBlock


class PackedModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.embedding = nn.Linear(5, 32)
        self.position = nn.Embedding(16, 32)
        self.temporal_layers = nn.ModuleList(
            [PreNormEncoderBlock(32, 4, 64, 0.) for _ in range(layers)])
        self.temporal_norm = nn.LayerNorm(32)

    def _embed_observation_rows(self, raw, mode, positions):
        return self.embedding(raw) + self.position(positions)


def cpu_varlen(q, k, v, cu_q, cu_k, max_q, max_k, *, window_size):
    assert window_size == (-1, 0)
    edges = cu_q.tolist()
    return torch.cat([
        F.scaled_dot_product_attention(q[a:b].transpose(0, 1),
            k[a:b].transpose(0, 1), v[a:b].transpose(0, 1),
            is_causal=True).transpose(0, 1)
        for a, b in zip(edges[:-1], edges[1:])])


@pytest.mark.parametrize('layers', [0, 1, 3])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_sparse_queries_match_independent_dense_histories_and_gradients(layers, device):
    if device == 'cuda' and (not torch.cuda.is_available() or not packed_attention.AVAILABLE):
        pytest.skip('requires CUDA variable-length attention')
    torch.set_num_threads(1)
    torch.manual_seed(1738)
    model = PackedModel(layers).to(device)
    reference = copy.deepcopy(model)
    raw = torch.randn(1, 13, 5, device=device, requires_grad=True)
    reference_raw = raw.detach().clone().requires_grad_(True)
    cu = torch.tensor([0, 4, 10, 13], dtype=torch.int32, device=device)
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2]], device=device)
    # Unsorted, repeated and zero-weight padding queries all occur in real graphs.
    queries = torch.tensor([12, 0, 8, 3, 4, 4, 9, 0], device=device)
    weights = torch.randn(8, 32, device=device)
    weights[-1].zero_()
    implementation = cpu_varlen if device == 'cpu' else packed_attention.varlen_attn
    with patch.object(packed_attention, 'varlen_attn', implementation), \
            torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
        actual = packed_attention.forward(model, raw, positions, cu, queries, 6, None)
        embedded = reference._embed_observation_rows(reference_raw, None, positions)
        sequences = []
        for a, b in [(0, 4), (4, 10), (10, 13)]:
            tokens = embedded[:, a:b]
            for layer in reference.temporal_layers:
                tokens = layer(tokens, valid_mask=torch.ones((1, b-a), dtype=torch.bool,
                                                             device=device), is_causal=True)
            sequences.append(tokens[0])
        expected = reference.temporal_norm(torch.cat(sequences).index_select(0, queries))
        actual_loss = (actual * weights).sum()
        expected_loss = (expected * weights).sum()
    tolerance = dict(rtol=.03, atol=.02) if device == 'cuda' else dict(rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual, expected, **tolerance)
    actual_loss.backward()
    expected_loss.backward()
    differences = []
    denominators = []
    for parameter, wanted in zip(model.parameters(), reference.parameters(), strict=True):
        assert (parameter.grad is None) == (wanted.grad is None)
        if wanted.grad is not None:
            if device == 'cpu':
                torch.testing.assert_close(parameter.grad, wanted.grad, rtol=3e-5, atol=3e-6)
            differences.append((parameter.grad-wanted.grad).square().sum())
            denominators.append(wanted.grad.square().sum())
    relative = (torch.stack(differences).sum()/torch.stack(denominators).sum()).sqrt()
    assert float(relative) < (.025 if device == 'cuda' else 2e-6)
    torch.testing.assert_close(raw.grad, reference_raw.grad, **tolerance)
