from tensorflow import keras as ks
import numpy as np
from gym.spaces import Box, Discrete


def space_n_to_shape_n(space_n):
    """
    Takes a list of gym spaces and returns a list of their shapes
    """
    return np.array([space_to_shape(space) for space in space_n])


def space_to_shape(space):
    """
    Takes a gym.space and returns its shape
    """
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [space.n]
    else:
        raise RuntimeError("Unknown space type. Can't return shape.")


def default_builder(structures):
    networks = {}
    for network in structures:
        name, struct = network['name'], network['struct']
        if type(struct) == str:
            networks[name] = ks.models.clone_model(networks[struct])
        else:
            inputs, outputs = None, None
            for layer in struct['layers']:
                t, p = layer['type'], layer['params']
                if t == 'input':
                    inputs = ks.Input(shape=(p['shape'],))
                    outputs = inputs
                else:
                    if t == 'dense':
                        outputs = ks.layers.Dense(units=p['units'], activation=p['activation'])(outputs)
                    else:
                        raise ValueError(f'Unknown layer type: {t}')
            networks[name] = ks.Model(inputs=inputs, outputs=outputs, name=name)
    return networks
