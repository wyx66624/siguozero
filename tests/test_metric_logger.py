"""Scalar event batching retains TensorBoard and durable JSONL contents."""
import json

import pytest

pytest.importorskip('tensorboard')
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from junqi.training.metrics import MetricLogger


def test_batched_events_preserve_every_tag_value_and_step(tmp_path):
    logger = MetricLogger(tmp_path)
    rows = [dict(loss=.125, count=17, enabled=True, phase='actor'),
            dict(loss=.0625, count=19, enabled=False, phase='critic')]
    for step, row in zip((7, 8), rows, strict=True):
        logger.log(step, row)
    logger.close()
    events = EventAccumulator(str(tmp_path/'tensorboard')).Reload()
    assert set(events.Tags()['scalars']) == {'loss','count','enabled'}
    for tag in ('loss','count','enabled'):
        assert [(event.step,event.value) for event in events.Scalars(tag)] == [
            (step,float(row[tag])) for step,row in zip((7,8),rows,strict=True)]
    durable = [json.loads(line) for line in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert [row['update'] for row in durable] == [7,8]
    for original, restored in zip(rows,durable,strict=True):
        assert all(restored[key] == value for key,value in original.items())
    assert json.loads((tmp_path/'latest_metrics.json').read_text()) == durable[-1]


def test_logging_with_no_numeric_metrics_keeps_jsonl(tmp_path):
    logger = MetricLogger(tmp_path)
    logger.log(3, {'phase':'collecting'})
    logger.close()
    assert json.loads((tmp_path/'metrics.jsonl').read_text())['phase'] == 'collecting'
    events = EventAccumulator(str(tmp_path/'tensorboard')).Reload()
    assert events.Tags()['scalars'] == []
