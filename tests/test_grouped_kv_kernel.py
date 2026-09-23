"""Direct attention must respect each game's live length during graph replay."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/Triton required')
@pytest.mark.parametrize('queries', [2, 4, 8, 16])
def test_mixed_lengths_and_changed_metadata_inside_cuda_graph(queries):
    from junqi.training.grouped_kv_kernel import attend_grouped
    torch.manual_seed(917)
    batch, heads, capacity, dim = 8, 8, 1001, 64
    q = torch.randn(batch, heads, queries, dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch, heads, capacity, dim, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    meta = torch.zeros(batch, 3, device='cuda', dtype=torch.long)
    def call():
        attend_grouped[(batch, heads)](q, k, v, meta, out, *q.stride()[:3],
            HEADS=heads, QUERIES=queries, DIM=dim, CAPACITY=capacity, EXTENT=1024)
    cases = [([0, 1, 63, 64, 127, 400, 700, 980], [0, 1, 2, queries, 1, queries, queries, queries]),
             ([500, 980, 0, 255, 64, 100, 1, 800], [queries, queries, queries, 0, queries, 2, 1, queries])]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2): call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream): call()
    for prefixes, counts in cases:
        meta.copy_(torch.tensor(list(zip(range(batch), prefixes, counts)), device='cuda'))
        graph.replay()
        expected = torch.zeros_like(q)
        for b, (prefix, count) in enumerate(zip(prefixes, counts, strict=True)):
            for row in range(count):
                end = prefix + row + 1
                scores = torch.einsum('hd,hld->hl',q[b,:,row].float(),k[b,:,:end].float()) / dim**.5
                expected[b,:,row] = torch.einsum('hl,hld->hd',scores.softmax(-1),v[b,:,:end].float())
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out,expected,atol=.012,rtol=.02)
