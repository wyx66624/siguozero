"""Small inference-only graphs for factorized categorical action sampling."""
import torch
from .cuda_graph_runtime import warmup_stream
from torch.nn import functional as F


class PendingActionSamples:
    """Own a pinned result until its stream-ordered device copy has completed.

    The sampling graph may replay or release its scratch storage before the
    caller asks for actions. Its output is copied now, on the same stream;
    only reading the independent host buffer is deferred.
    """

    def __init__(self, sampled, *, return_log_probs):
        self.host = torch.empty(sampled.shape, dtype=sampled.dtype, device='cpu', pin_memory=True)
        self.host.copy_(sampled, non_blocking=True)
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(sampled.device))
        self.return_log_probs = return_log_probs
        self._result = None

    def result(self):
        if self._result is None:
            self.ready.synchronize()
            pairs = self.host[:, :2].to(torch.long).tolist()
            logs = list(self.host[:, 2].split(1)) if self.return_log_probs else []
            self._result = ([[tuple(pair)] for pair in pairs], logs)
        return self._result


def sample_tensor(model, contexts, source_masks, destination_masks, uniforms, temperature):
    from .models import _categorical_from_uniforms
    points = source_masks.shape[1]
    rows = torch.arange(contexts.shape[0], device=contexts.device)
    source_logs = F.log_softmax((model.source_query(contexts)[:, :points] / temperature)
                                .masked_fill(~source_masks, float('-inf')), dim=-1)
    sources = _categorical_from_uniforms(source_logs.exp(), uniforms[:, :1]).squeeze(-1)
    logits = model._destination_logits(contexts, sources, points) / temperature
    destination_logs = F.log_softmax(logits.masked_fill(~destination_masks[rows, sources], float('-inf')), dim=-1)
    destinations = _categorical_from_uniforms(destination_logs.exp(), uniforms[:, 1:]).squeeze(-1)
    logs = source_logs[rows, sources] + destination_logs[rows, destinations]
    return torch.stack((sources.float(), destinations.float(), logs.float()), dim=1)


def sample_graph(model, contexts, source_masks, destination_masks, uniforms, temperature):
    key = (tuple(contexts.shape), contexts.dtype, tuple(source_masks.shape), temperature,
           torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else None)
    entry = model._ppo_sampling_graphs.get(key)
    if entry is None:
        inputs = [x.clone() for x in (contexts, source_masks, destination_masks, uniforms)]
        stream = warmup_stream(contexts.device)
        stream.wait_stream(torch.cuda.current_stream(contexts.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                sample_tensor(model, *inputs, temperature)
        torch.cuda.current_stream(contexts.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # Uniforms are inputs. Capturing/warming consumes no random draws.
        with torch.cuda.graph(graph):
            output = sample_tensor(model, *inputs, temperature)
        entry = graph, inputs, output
        model._ppo_sampling_graphs[key] = entry
        while len(model._ppo_sampling_graphs) > 8:
            model._ppo_sampling_graphs.popitem(last=False)
    else:
        model._ppo_sampling_graphs.move_to_end(key)
    graph, inputs, output = entry
    for target, source in zip(inputs, (contexts, source_masks, destination_masks, uniforms), strict=True):
        target.copy_(source)
    graph.replay()
    return output
