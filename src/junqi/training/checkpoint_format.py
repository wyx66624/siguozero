"""Checkpoint architecture contract, importable without the torch runtime."""

# Version 8 adds a sixth action input and a shared draw-value critic head.
# Versions 6/7 retain every learned input weight and optimizer moment; the
# added column/head start at zero. Version 7 also expanded the capture counter.
# Version 9 appends four public pass budgets to the board projection.
CHECKPOINT_FORMAT_VERSION = 9
# Version 10 records the expanded concat-token SwiGLU architecture. Legacy
# profiles still write v9; accepting old inference files does not migrate them.
EXPAND_CHECKPOINT_FORMAT_VERSION = 10
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = (6, 7, 8, 9, 10)


def checkpoint_format_for_model(model):
    fusion = (model.get("token_fusion_hidden_dim", 0) if isinstance(model, dict)
              else getattr(model, "token_fusion_hidden_dim", 0))
    return EXPAND_CHECKPOINT_FORMAT_VERSION if fusion else CHECKPOINT_FORMAT_VERSION
