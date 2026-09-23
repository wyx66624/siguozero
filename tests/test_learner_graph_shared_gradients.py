"""CUDA graph shapes reuse buffers without retaining an earlier gradient."""
import pytest
import torch

from junqi.training.learner_graph import PackedLearnerGraph


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs required')
def test_shapes_share_gradients_and_preserve_unused_parameters():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)
            self.unused = torch.nn.Parameter(torch.ones(4))
            self.device = torch.device('cuda')
        def _packed_temporal_forward(self, values):
            return self.linear(values)
    torch.manual_seed(917)
    model = Model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
    pool, cache = torch.cuda.graph_pool_handle(), {}
    inputs = [torch.randn(n,3,device='cuda') for n in (4,7)]
    graphs=[]
    for values in inputs:
        graph=PackedLearnerGraph(model,critic=True,amp_dtype=torch.bfloat16,
            pool=pool,gradient_cache=cache)
        def loss(current, *, values=values, graph=graph):
            result=graph.callable(values).float().square().mean()
            return result,torch.stack([result.detach()]*7)
        graph._loss=loss
        graphs.append(graph)
    for index in [0,1,0,1]:
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            expected_loss=model.linear(inputs[index]).float().square().mean()
        expected_loss.backward()
        expected=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
        # Release the default-stream oracle's autograd nodes before capture
        # creates its AccumulateGrad nodes on the dedicated warmup stream.
        del expected_loss
        graphs[index].run(model)
        for parameter,reference in zip(model.parameters(),expected,strict=True):
            assert (parameter.grad is None)==(reference is None)
            if reference is not None:
                torch.testing.assert_close(parameter.grad,reference,atol=.003,rtol=.02)
        optimizer.step()
        torch.testing.assert_close(model.unused,torch.ones_like(model.unused),atol=0,rtol=0)
    addresses=[tuple(None if g is None else g.data_ptr() for g in graph.grads) for graph in graphs]
    assert addresses[0]==addresses[1]
