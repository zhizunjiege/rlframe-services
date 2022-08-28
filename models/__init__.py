from .ddqn import DDQN
from .dqn import DQN
from .vpg import VPG

RLModels = {
    'DDQN': DDQN,
    'DQN': DQN,
    'VPG': VPG,
}


def default_builder(structures):
    ...
