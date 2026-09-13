"""Checkpoint architecture contract, importable without the torch runtime."""

# Version 8 adds a sixth action input and a shared draw-value critic head.
# Versions 6/7 retain every learned input weight and optimizer moment; the
# added column/head start at zero. Version 7 also expanded the capture counter.
# Version 9 appends four public pass budgets to the board projection.
CHECKPOINT_FORMAT_VERSION = 9
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = (6, 7, 8, 9)
