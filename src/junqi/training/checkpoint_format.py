"""Checkpoint architecture contract, importable without the torch runtime."""

from ..inventory import INVENTORY_ENCODER_TYPE

# Versions 15/16 reserve confirmed Actor slots exclusively, removing them
# from other same-owner candidates. Earlier masks have different semantics.
INVENTORY_CHECKPOINT_FORMAT_VERSION = 15
CHECKPOINT_FORMAT_VERSION = INVENTORY_CHECKPOINT_FORMAT_VERSION
# Version 16 also folds countdown/alive state into the action projection and removes
# the separate countdown, alive and revealed-flag embeddings.
ACTION_CONTEXT_CHECKPOINT_FORMAT_VERSION = 16
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = (15, 16)


def checkpoint_format_for_model(model):
    encoder = (model.get("board_encoder_type") if isinstance(model, dict)
               else getattr(model, "board_encoder_type", None))
    if encoder != INVENTORY_ENCODER_TYPE:
        raise ValueError("checkpoint requires inventory multihot board encoding; legacy encoding is unsupported")
    action = (model.get("action_encoder_type") if isinstance(model, dict)
              else getattr(model, "action_encoder_type", None))
    if action == "coordinates_player_countdown_active_linear":
        return ACTION_CONTEXT_CHECKPOINT_FORMAT_VERSION
    return INVENTORY_CHECKPOINT_FORMAT_VERSION
