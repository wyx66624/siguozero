"""Exclusive GPU check of bounded cold prefill and a complete sliding window."""
import fcntl
import json
from dataclasses import replace
from pathlib import Path
import tempfile

import torch

from benchmark_ppo import diverse_states
from junqi.training.history_arrays import ArrayHistory
from junqi.training.models import GamePolicyTransformer
from junqi.training.settings import TrainingSettings


def main():
    lock = open(Path(tempfile.gettempdir()) / 'siguozero-cuda-probe.lock', 'a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.90)
    settings = TrainingSettings.from_yaml('configs/local_4090_training.yaml', 'four_dark', model_scale='main')
    checkpoint = Path('output/throughput_20260913/checkpoint_1336.pt')
    before = checkpoint.stat()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    model = GamePolicyTransformer(settings.model).cuda().eval()
    model.load_state_dict(payload['policy'])
    templates = diverse_states(settings.mode, settings.model, 1001, 8)
    histories, states = [], []
    for i in range(64):
        template = templates[i % len(templates)]
        length = (1, 128, 256, 512, 1001)[i % 5]
        history = ArrayHistory(settings.mode, template.records[:length], 1000, (i, 0))
        histories.append(history)
        states.append(replace(template, records=history.view()))
    model.start_ppo_inference_cache(capacity=512, behavior_version=1336)
    store = model._fixed_kv_store
    results = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for iteration in range(2):
            if iteration:
                for i, history in enumerate(histories):
                    if i % 2 == 0:
                        history.append(templates[i % len(templates)].records[-1])
                states = [replace(state, records=history.view())
                          for state, history in zip(states, histories, strict=True)]
            model._fixed_kv_store = None
            expected = torch.cat([model._encode_full(states[i:i+4], pack_prefixes=True).context
                                  for i in range(0, len(states), 4)])
            model._fixed_kv_store = store
            actual = model.encode(states).context
            difference = actual.float() - expected.float()
            relative = float(difference.norm() / expected.float().norm())
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=.04, atol=.04)
            assert relative < .025
            torch.testing.assert_close(model.encode(states).context, actual, rtol=0, atol=0)
            assert store.prefill_max_padded_tokens <= store.PREFILL_PADDED_TOKENS
            results.append(dict(iteration=iteration, states=len(states), full_window=max(len(s.records) for s in states),
                maximum_context_error=float(difference.abs().max()), relative_context_l2=relative,
                max_prefill_padded_tokens=store.prefill_max_padded_tokens,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    assert (before.st_size, before.st_mtime_ns) == (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
    path = Path('docs/benchmarks/throughput_bounded_prefill_validation_4090_20260913.json')
    path.write_text(json.dumps(dict(checkpoint_update=1336, cache_capacity=512, results=results), indent=2)+'\n')
    print(path, json.dumps(results))


if __name__ == '__main__':
    main()
