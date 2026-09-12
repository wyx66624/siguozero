"""Read each cached K/V tile once for all pending queries of a player."""
import triton
import triton.language as tl


@triton.jit
def attend_grouped(Q, CACHE_K, CACHE_V, META, OUT, QS0, QS1, QS2,
                   HEADS: tl.constexpr, QUERIES: tl.constexpr, DIM: tl.constexpr,
                   CAPACITY: tl.constexpr, EXTENT: tl.constexpr,
                   BLOCK_N: tl.constexpr = 64, BLOCK_M: tl.constexpr = 16):
    batch, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(META + batch * 3)
    prefix = tl.load(META + batch * 3 + 1)
    count = tl.load(META + batch * 3 + 2)
    rows, dims = tl.arange(0, BLOCK_M), tl.arange(0, DIM)
    valid_queries = (rows < QUERIES) & (rows < count)
    q = tl.load(Q + batch * QS0 + head * QS1 + rows[:, None] * QS2 + dims[None, :],
                valid_queries[:, None], 0)
    accumulator = tl.full((BLOCK_M, DIM), 0., tl.float32)
    normalizer = tl.full((BLOCK_M,), 0., tl.float32)
    maximum = tl.full((BLOCK_M,), -1.0e6, tl.float32)
    base = (slot * HEADS + head) * CAPACITY * DIM
    for start in range(0, EXTENT, BLOCK_N):
        times = start + tl.arange(0, BLOCK_N)
        valid_keys = (times < prefix + count) & (times < CAPACITY)
        k = tl.load(CACHE_K + base + times[None, :] * DIM + dims[:, None], valid_keys[None, :], 0)
        v = tl.load(CACHE_V + base + times[:, None] * DIM + dims[None, :], valid_keys[:, None], 0)
        scores = tl.dot(q, k) * (DIM ** -0.5)
        allowed = valid_keys[None, :] & (times[None, :] <= prefix + rows[:, None]) & valid_queries[:, None]
        scores = tl.where(allowed, scores, -1.0e6)
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp(maximum - next_maximum)
        probabilities = tl.exp(scores - next_maximum[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(v.dtype), v)
        maximum = next_maximum
    output = tl.where(valid_queries[:, None], accumulator / normalizer[:, None], 0.)
    tl.store(OUT + ((batch * HEADS + head) * QUERIES + rows[:, None]) * DIM + dims[None, :],
              output, (rows < QUERIES)[:, None])
