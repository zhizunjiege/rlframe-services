from .dqn import DQN
from .double_dqn import DoubleDQN

from .ddpg import DDPG

RLModels = {
    'DQN': DQN,
    'DoubleDQN': DoubleDQN,
    'DDPG': DDPG,
}
