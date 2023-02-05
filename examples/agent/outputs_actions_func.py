from typing import Any, Dict, List

import numpy as np


def func(outputs: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
    """Convert `outputs` to `actions` for model simulation."""
    return {
        'example_uav': [{
            'azimuth': float(45 * outputs),
        }],
    }
