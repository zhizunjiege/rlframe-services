from tensorflow import keras as ks

from .dqn import DQN

RLModels = {
    'DQN': DQN,
}


def default_builder(structures):
    networks = {}
    for name, struct in structures.items():
        if type(struct) == str:
            struct = structures[struct]
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
