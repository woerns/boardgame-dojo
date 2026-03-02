"""Core type aliases and sentinel values for the dojo framework."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# A player is identified by a non-negative integer (0, 1, ...).
PlayerId = int

# Actions are flat integers indexing into the policy vector.
Action = int

# Boolean mask over the action space — True means legal.
ActionMask = npt.NDArray[np.bool_]

# Observation tensor fed to the neural network.
ObservationTensor = npt.NDArray[np.float32]

# Sentinel player IDs for non-decision nodes.
CHANCE_PLAYER: PlayerId = -1
SIMULTANEOUS_PLAYER: PlayerId = -2
TERMINAL_PLAYER: PlayerId = -3
