"""Compare grouped KV reads with the original kernel and exact causal SDPA."""
import argparse
import fcntl
import json
from pathlib import Path
import tempfile

import torch
import triton

from junqi.training.fixed_kv_kernels import attend_kv
from junqi.training.grouped_kv_kernel import attend_grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16')
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    lock = open(Path(tempfile.gettempdir()) / 'siguozero-cuda-probe.lock', 'a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(913)
    rows = []
    dtype = getattr(torch, args.dtype)
    for batch in (8, 48):
        for count in (1, 4, 8):
            for extent in (64, 512, 1024):
                capacity, heads, dim = 1001, 8, 32
                k = torch.randn(batch, heads, capacity, dim, device='cuda', dtype=dtype)
                v = torch.randn_like(k)
                q = torch.randn(batch, heads, count, dim, device='cuda', dtype=dtype)
                meta = torch.tensor([(i, min(capacity, extent) - count - i % 7, count - i % count)
                                     for i in range(batch)], device='cuda')
                meta[-1, 2] = 0
                a, b = torch.empty_like(q), torch.empty_like(q)
                def old():
                    attend_kv[(batch, heads, count)](q, k, v, meta, a, *q.stride()[:3],
                        HEADS=heads, QUERIES=count, DIM=dim, CAPACITY=capacity,
                        BLOCK_N=extent, BLOCK_D=dim, SCALE=dim ** -.5, num_warps=4 if extent <= 512 else 8)
                def new():
                    attend_grouped[(batch, heads)](q, k, v, meta, b, *q.stride()[:3],
                        HEADS=heads, QUERIES=count, DIM=dim, CAPACITY=capacity, EXTENT=extent)
                old()
                new()
                torch.testing.assert_close(a, b, atol=.015, rtol=.015)
                assert torch.equal(b[-1], torch.zeros_like(b[-1]))
                times = torch.arange(capacity, device='cuda')[None, None, :]
                queries = torch.arange(count, device='cuda')[None, :, None]
                allowed = ((times < (meta[:, 1] + meta[:, 2])[:, None, None])
                           & (times <= meta[:, 1, None, None] + queries)
                           & (queries < meta[:, 2, None, None]))
                reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])
                torch.testing.assert_close(b, reference, atol=.015, rtol=.015)
                old_ms = None if args.validation_only else triton.testing.do_bench(old, warmup=50, rep=100)
                new_ms = None if args.validation_only else triton.testing.do_bench(new, warmup=50, rep=100)
                rows.append(dict(batch=batch, queries=count, extent=extent, old_ms=old_ms,
                                 grouped_ms=new_ms, speedup=None if args.validation_only else old_ms/new_ms,
                                 max_absolute_error=float((a-b).abs().max())))
                rows[-1]['sdpa_max_absolute_error'] = float((b-reference).abs().max())
                print(json.dumps(rows[-1]), flush=True)
    Path(args.output).write_text(json.dumps(dict(complete=True, dtype=args.dtype, cases=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
