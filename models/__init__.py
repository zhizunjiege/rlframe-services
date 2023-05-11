from .dqn import DQN
from .double_dqn import DoubleDQN

from .ddpg import DDPG
from .maddpg import MADDPG

RLModels = {
    'DQN': DQN,
    'DoubleDQN': DoubleDQN,
    'DDPG': DDPG,
    'MADDPG': MADDPG,
}
