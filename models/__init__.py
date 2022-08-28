from .dqn import DQN

RLModels = {
    'DQN': DQN,
}


def default_builder(structures):
    ...
