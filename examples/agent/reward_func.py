from typing import Any, Dict, List

import numpy as np


def func(
    states: Dict[str, List[Dict[str, Any]]],
    inputs: np.ndarray,
    actions: Dict[str, List[Dict[str, Any]]],
    outputs: np.ndarray,
    next_states: Dict[str, List[Dict[str, Any]]],
    next_inputs: np.ndarray,
    terminated: bool,
    truncated: bool,
) -> float:
    """Calculate the reward for the current step."""
    if terminated:
        return 1.0
    else:
        return 0.0
