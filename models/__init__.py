from .dqn import DQN
from .double_dqn import DoubleDQN
from .qmix_latest.qmix import QMIX
from .ddpg import DDPG
from .maddpg import MADDPG

RLModels = {model.name: model for model in [
    DQN,
    DoubleDQN,
    DDPG,
    MADDPG,
    QMIX
]}
