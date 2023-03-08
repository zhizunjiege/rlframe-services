from typing import Any, Dict, List

import numpy as np


def func(outputs: np.ndarray | Dict[str | int, np.ndarray]) -> Dict[str, List[Dict[str, Any]]]:
    """Convert `outputs` to `actions` for model simulation."""
    return {
        'example_uav': [{
            'azimuth': float(45 * outputs),
            'example_struct': {
                'field1': False,
                'field2': 0,
                'field3': 0,
                'field4': 0.0,
                'field5': '',
            },
            'example_array': [],
            'example_combine': [],
            'example_nest': {
                'field1': False,
                'field2': [],
                'field3': {
                    'field1': False,
                    'field2': 0,
                    'field3': 0,
                    'field4': 0.0,
                    'field5': '',
                },
                'field4': [],
            },
        }],
    }
