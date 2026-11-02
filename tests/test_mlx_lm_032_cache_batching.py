"""Exercise the mlx-lm 0.32 cache protocol used by continuous batching."""
import importlib

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

from vmlx_engine.utils.mamba_cache import ensure_mamba_support


def test_hybrid_cache_merge_preserves_empty_ssm_rows():
    ensure_mamba_support()
    generate = importlib.import_module("mlx_lm.generate")
    caches = []
    for value in (3.0, 7.0):
        kv = KVCache()
        tensor = mx.full((1, 1, 2, 2), value)
        kv.update_and_fetch(tensor, tensor)
        ssm = ArraysCache(2)
        if value == 7.0:
            ssm[0] = mx.full((1, 2), value)
            ssm[1] = mx.full((1, 3), value + 1)
        caches.append([kv, ssm])

    attention, state = generate._merge_caches(caches)
    assert isinstance(attention, BatchKVCache)
    assert isinstance(state, ArraysCache)
    assert state[0].tolist() == [[0.0, 0.0], [7.0, 7.0]]
    assert state[1].tolist() == [[0.0, 0.0, 0.0], [8.0, 8.0, 8.0]]
    for index, value in enumerate((3.0, 7.0)):
        extracted = attention.extract(index)
        assert extracted.keys.tolist() == mx.full((1, 1, 2, 2), value).tolist()
