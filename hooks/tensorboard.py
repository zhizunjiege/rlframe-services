from collections.abc import Iterable
from datetime import datetime

from tensorboardX import SummaryWriter

from models.base import RLModelBase

from .base import HookBase, AnyDict


class Tensorboard(HookBase):
    """Auto log to tensorboard."""

    def __init__(self, model: RLModelBase):
        """Init hook."""
        super().__init__(model)

        self.logdir = f'data/logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.writer = SummaryWriter(logdir=self.logdir)

        self.rewards, self.returns = {}, {}
        self.step = 0

    def react2train(self, step: int, rewargs: AnyDict, caches: AnyDict):
        """Before train."""
        reward = rewargs['reward']

        if isinstance(reward, dict):
            for k, v in reward.items():
                self.rewards.setdefault(k, []).append(v)
        else:
            self.rewards.setdefault('#', []).append(reward)

    def after_train(self, step: int, infos: AnyDict):
        """After train."""
        for k, v in infos.items():
            self.writer.add_scalar(f'train/{k}', sum(v) / len(v) if isinstance(v, Iterable) else v, step)

        self.step = step

    def after_episode(self, episode: int):
        """After episode."""
        if self.model.training:
            self.returns.clear()

        for k, v in self.rewards.items():
            self.returns.setdefault(k, []).append(self._calc_return(self.model.gamma, v))
        self.rewards.clear()

        title = 'train/return' if self.model.training else 'test/return'
        if len(self.returns) > 1:
            for k, v in self.returns.items():
                self.writer.add_scalar(f'{title}/{k}', sum(v) / len(v), self.step)
        elif len(self.returns) == 1:
            self.writer.add_scalar(title, sum(self.returns['#']) / len(self.returns['#']), self.step)

    def __del__(self):
        """Close hook."""
        self.writer.close()
        super().__del__()

    def _calc_return(self, gamma, rewards):
        g = 0
        for r in reversed(rewards):
            g = r + gamma * g
        return g
