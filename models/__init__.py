from .dqn import DQN
from .double_dqn import DoubleDQN

from .ddpg import DDPG
from .maddpg import MADDPG

RLModels = {model.__name__: model for model in [
    DQN,
    DoubleDQN,
    DDPG,
    MADDPG,
]}
