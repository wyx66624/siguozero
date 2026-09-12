"""CUDA inference kernels for direct access to fixed game/seat KV slots."""

import triton
import triton.language as tl


@triton.jit
def write_kv(K, V, CACHE_K, CACHE_V, META,
             BATCH, TOKENS, KS0, KS1, KS2, VS0, VS1, VS2,
             HEADS: tl.constexpr, DIM: tl.constexpr, CAPACITY: tl.constexpr,
             BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dimension = index % DIM
    token = (index // DIM) % TOKENS
    head = (index // (DIM * TOKENS)) % HEADS
    batch = index // (DIM * TOKENS * HEADS)
    slot = tl.load(META + batch * 3, batch < BATCH, 0)
    prefix = tl.load(META + batch * 3 + 1, batch < BATCH, 0)
    count = tl.load(META + batch * 3 + 2, batch < BATCH, 0)
    valid = (batch < BATCH) & (token < count) & (prefix + token < CAPACITY)
    keys = tl.load(K + batch * KS0 + head * KS1 + token * KS2 + dimension, valid, 0)
    values = tl.load(V + batch * VS0 + head * VS1 + token * VS2 + dimension, valid, 0)
    destination = ((slot * HEADS + head) * CAPACITY + prefix + token) * DIM + dimension
    tl.store(CACHE_K + destination, keys, valid)
    tl.store(CACHE_V + destination, values, valid)


@triton.jit
def attend_kv(Q, CACHE_K, CACHE_V, META, OUT,
              QS0, QS1, QS2,
              HEADS: tl.constexpr, QUERIES: tl.constexpr, DIM: tl.constexpr,
              CAPACITY: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
              SCALE: tl.constexpr):
    batch, head, query = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(META + batch * 3)
    prefix = tl.load(META + batch * 3 + 1)
    count = tl.load(META + batch * 3 + 2)
    times = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    valid_query = query < count
    valid_keys = (times < prefix + count) & (times < CAPACITY)
    pointer = ((slot * HEADS + head) * CAPACITY + times[:, None]) * DIM + dims[None, :]
    keys = tl.load(CACHE_K + pointer, valid_keys[:, None] & (dims[None, :] < DIM), 0).to(tl.float32)
    q = tl.load(Q + batch * QS0 + head * QS1 + query * QS2 + dims, dims < DIM, 0).to(tl.float32)
    scores = tl.sum(keys * q[None, :], axis=1) * SCALE
    allowed = valid_keys & (times <= prefix + query)
    scores = tl.where(allowed & valid_query, scores, -float("inf"))
    # Dummy rows must remain finite without reading or writing another slot.
    scores = tl.where(~valid_query & (times == 0), 0., scores)
    probabilities = tl.exp(scores - tl.max(scores, axis=0))
    probabilities = probabilities / tl.sum(probabilities, axis=0)
    values = tl.load(CACHE_V + pointer, valid_keys[:, None] & (dims[None, :] < DIM), 0).to(tl.float32)
    result = tl.sum(probabilities[:, None] * values, axis=0)
    result = tl.where(valid_query, result, 0.)
    tl.store(OUT + ((batch * HEADS + head) * QUERIES + query) * DIM + dims, result, dims < DIM)
