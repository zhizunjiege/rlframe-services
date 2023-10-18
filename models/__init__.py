from .dqn import DQN
from .doubledqn import DoubleDQN
from .ddpg import DDPG
from .ppo import PPO
from .qmix_latest.qmix import QMIX
from .maddpg import MADDPG

RLModels = {model.__name__: model for model in [
    DQN,
    DoubleDQN,
    DDPG,
    PPO,
    MADDPG,
    QMIX
]}
