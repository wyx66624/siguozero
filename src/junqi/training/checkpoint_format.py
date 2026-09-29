"""Checkpoint architecture contract, importable without the torch runtime."""

from ..inventory import INVENTORY_ENCODER_TYPE, INVENTORY_ONLY_ENCODER_TYPE

# Versions 15/16 reserve confirmed Actor slots exclusively, removing them
# from other same-owner candidates. Earlier masks have different semantics.
INVENTORY_CHECKPOINT_FORMAT_VERSION = 15
CHECKPOINT_FORMAT_VERSION = INVENTORY_CHECKPOINT_FORMAT_VERSION
# Version 16 also folds countdown/alive state into the action projection and removes
# the separate countdown, alive and revealed-flag embeddings.
ACTION_CONTEXT_CHECKPOINT_FORMAT_VERSION = 16
# Version 17 removes 75 casualty + 4 pass board inputs, and adds passes to actions.
CANDIDATE_CONTEXT_CHECKPOINT_FORMAT_VERSION = 17
# Version 18 is the 60-point, candidate-only two-player model without pass inputs.
TWO_PLAYER_NO_PASS_CHECKPOINT_FORMAT_VERSION = 18
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = (15, 16, 17, 18)


def checkpoint_format_for_model(model):
    encoder = (model.get("board_encoder_type") if isinstance(model, dict)
               else getattr(model, "board_encoder_type", None))
    if encoder not in (INVENTORY_ENCODER_TYPE, INVENTORY_ONLY_ENCODER_TYPE):
        raise ValueError("checkpoint requires inventory multihot board encoding; legacy encoding is unsupported")
    action = (model.get("action_encoder_type") if isinstance(model, dict)
              else getattr(model, "action_encoder_type", None))
    points = (model.get("board_point_capacity", 129) if isinstance(model, dict)
              else getattr(model, "board_point_capacity", 129))
    if (encoder == INVENTORY_ONLY_ENCODER_TYPE and points == 60
            and action == "coordinates_player_countdown_active_linear"):
        return TWO_PLAYER_NO_PASS_CHECKPOINT_FORMAT_VERSION
    if encoder == INVENTORY_ONLY_ENCODER_TYPE or action == "coordinates_player_countdown_active_passes_linear":
        if (encoder != INVENTORY_ONLY_ENCODER_TYPE
                or action != "coordinates_player_countdown_active_passes_linear"):
            raise ValueError("candidate-only board encoding requires the action encoder with pass counts")
        return CANDIDATE_CONTEXT_CHECKPOINT_FORMAT_VERSION
    if action == "coordinates_player_countdown_active_linear":
        return ACTION_CONTEXT_CHECKPOINT_FORMAT_VERSION
    return INVENTORY_CHECKPOINT_FORMAT_VERSION
