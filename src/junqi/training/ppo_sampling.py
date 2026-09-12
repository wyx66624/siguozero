"""Small inference-only graphs for factorized categorical action sampling."""
import torch
from torch.nn import functional as F


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
        stream = torch.cuda.Stream(device=contexts.device)
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
