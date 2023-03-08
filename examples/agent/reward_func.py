from typing import Any, Dict, List

import numpy as np


def func(
    states: Dict[str, List[Dict[str, Any]]],
    inputs: np.ndarray | Dict[str | int, np.ndarray],
    actions: Dict[str, List[Dict[str, Any]]],
    outputs: np.ndarray | Dict[str | int, np.ndarray],
    next_states: Dict[str, List[Dict[str, Any]]],
    next_inputs: np.ndarray | Dict[str | int, np.ndarray],
    terminated: bool,
    truncated: bool,
) -> float | Dict[str | int, float]:
    """Calculate the reward for the current step."""
    if terminated:
        return 1.0
    else:
        return 0.0
