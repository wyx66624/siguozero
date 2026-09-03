"""PyTorch self-play training and inference for SiguoZero.

PyTorch is an optional dependency of the rules package.  Install the
``training`` extra before importing model or trainer modules.
"""

from .modes import MODE_SPECS, ModeSpec, TrainingMode

__all__ = ["MODE_SPECS", "ModeSpec", "TrainingMode"]
