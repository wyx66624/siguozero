"""Project legacy two-player action weights from 14 inputs to 10, without RNG use.

This drops the four remaining-pass columns. It is exactly the old projection
with all pass inputs zero. Immutable historical checkpoints stay unchanged on disk.
"""
from .checkpoint_format import checkpoint_format_for_model
from .encoding import CONTEXT_ACTION_ENCODER_TYPE, PASS_CONTEXT_ACTION_ENCODER_TYPE


WEIGHT = "action_encoder.projection.weight"


def remove_two_player_pass_inputs(payload, *, optimizer_modules=None):
    model = payload.get("config", {}).get("model", {})
    if payload.get("mode") != "two_player" or payload.get("format_version") != 17:
        return False
    if model.get("board_point_capacity") != 60 or model.get("action_encoder_type") != PASS_CONTEXT_ACTION_ENCODER_TYPE:
        raise ValueError("unsupported two-player pass-feature migration")
    updates = []
    for name in ("policy", "critic", "reference_policy"):
        weights = payload.get(name)
        if weights is None:
            continue
        weight = weights[WEIGHT]
        if weight.ndim != 2 or weight.shape[1] != 14:
            raise ValueError("legacy action projection must have 14 input columns")
        updates.append((weights, WEIGHT, weight[:, :10].contiguous()))
        optimizer = payload.get(name + "_optimizer")
        if optimizer is not None and optimizer_modules is not None:
            module = optimizer_modules[name]
            names = [n for n, p in module.named_parameters()]
            ids = [i for group in optimizer["param_groups"] for i in group["params"]]
            if len(ids) != len(names):
                raise ValueError("optimizer parameter ordering changed during no-pass migration")
            moments = optimizer["state"].get(ids[names.index(WEIGHT)], {})
            for key, value in moments.items():
                if getattr(value, "ndim", 0) == 0:
                    continue
                if value.shape != weight.shape:
                    raise ValueError("unexpected action optimizer moment shape")
                updates.append((moments, key, value[:, :10].contiguous()))
    packed = payload.get("packed_weights", {}).get("policy")
    if packed is not None:
        from .tactical_opponent import pack_state_dict
        state = {name: pack["flat"][start:end].view(shape)
                 for pack in packed.values() for name, (start, end, shape) in pack["views"].items()}
        weight = state[WEIGHT]
        if weight.ndim != 2 or weight.shape[1] != 14:
            raise ValueError("legacy packed action projection must have 14 input columns")
        state[WEIGHT] = weight[:, :10].contiguous()
        updates.append((payload["packed_weights"], "policy", pack_state_dict(state)))
    for container, key, value in updates:
        container[key] = value
    model["action_encoder_type"] = CONTEXT_ACTION_ENCODER_TYPE
    payload["format_version"] = checkpoint_format_for_model(model)
    payload["config"]["max_passes_per_player"] = 0
    return True
