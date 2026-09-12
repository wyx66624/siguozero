"""Persist the final PPO optimization validation evidence and its hashes."""
import hashlib
import json
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parents[1]
    test_log = root/'output/ppo20_full_tests_final_20260911.log'
    log = test_log.read_text()
    run = re.search(r'Ran (\d+) tests in ([\d.]+)s', log)
    success = re.search(r'\nOK \(skipped=(\d+)\)\s*$', log)
    assert run and success, 'the final complete unittest run did not pass'
    total, skipped = int(run[1]), int(success[1])
    names = (
        'ppo20_varlen_validation_4090_20260911.json',
        'ppo20_bf16_residual_validation_4090_20260911.json',
        'ppo20_fused_sampling_verified_4090_20260911.json',
        'ppo20_kv_fp16_sdpa_20260911.json',
        'ppo20_kv_bf16_sdpa_20260911.json',
        'ppo20_capacity_4090_20260911.json',
        'ppo20_ddp_resume_20260911.json',
        'ppo20_default48_final_4090_20260911.json',
        'ppo20_batch1024_4090_20260911.json',
        'ppo20_final_eta_20260911.json',
    )
    records = []
    for name in names:
        path = root/'docs/benchmarks'/name
        content = path.read_bytes()
        result = json.loads(content.decode('utf-8-sig'))
        assert result.get('complete') and not result.get('candidate_validation_error'), name
        records.append(dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(content).hexdigest(), complete=True))
    result = dict(complete=True, tests=dict(total=total, passed=total-skipped, skipped=skipped,
        seconds=float(run[2]), log=str(test_log.relative_to(root)),
        sha256=hashlib.sha256(test_log.read_bytes()).hexdigest()),
        artifacts=records,
        validation_source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root/'src/junqi').rglob('*.py'))},
        limitations=['CPU/Gloo verifies distributed behavior; multi-GPU CUDA throughput was not measured.',
                     'GPU numerical tests complement the CPU suite; this is not a long-run strength evaluation.',
                     'The twenty-day time target remains unmet.'])
    output = root/'docs/benchmarks/ppo20_validation_summary_20260911.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['tests']))


if __name__ == '__main__':
    main()
