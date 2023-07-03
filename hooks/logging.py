from collections.abc import Iterable
from datetime import datetime
import logging
import os

from tensorboardX import SummaryWriter

from models.base import RLModelBase

from .base import HookBase, AnyDict, LOGGER_NAME


class Logging(HookBase):
    """Auto logging to terminal and tensorboard."""

    def __init__(self, model: RLModelBase, *, loglvl='INFO', terminal=True, tensorboard=True):
        """Init hook.

        Args:
            model: RLModel instance.
            loglvl: Logging level.
            terminal: Whether to log to terminal.
            tensorboard: Whether to log to tensorboard.
        """
        super().__init__(model)

        self.loglvl = loglvl.upper()
        self.terminal = terminal
        self.tensorboard = tensorboard

        self.logdir = f'data/logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        os.makedirs(self.logdir, exist_ok=True)

        if self.terminal:
            self.handler = logging.FileHandler(f'{self.logdir}/log.txt', mode="w", encoding='utf-8')
            self.handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger = logging.getLogger(LOGGER_NAME)
            self.logger.addHandler(self.handler)
            self.logger.setLevel(self.loglvl)
        if self.tensorboard:
            self.writer = SummaryWriter(logdir=self.logdir)

        self.rewards, self.returns = {}, {}
        self.step = 0

    def after_react(self, step: int, siargs: AnyDict, oaargs: AnyDict):
        if self.terminal:
            self.logger.debug(f'React step {step} ended, states: {siargs["states"]}, actions: {oaargs["actions"]}.')

    def react2train(self, rewargs: AnyDict):
        reward = rewargs['reward']

        if isinstance(reward, dict):
            for k, v in reward.items():
                self.rewards.setdefault(k, []).append(v)
        else:
            self.rewards.setdefault('#', []).append(reward)

        if self.terminal:
            self.logger.debug(f'React to Train: reward: {reward}.')

    def after_train(self, step: int, infos: AnyDict):
        self.step = step

        if self.terminal:
            self.logger.debug(f'Train step {step} ended, infos: {infos}.')
        if self.tensorboard:
            for k, v in infos.items():
                self.writer.add_scalar(f'train/{k}', sum(v) / len(v) if isinstance(v, Iterable) else v, step)

    def after_episode(self, episode: int, shared: AnyDict):
        if self.model.training:
            self.returns.clear()

        for k, v in self.rewards.items():
            self.returns.setdefault(k, []).append(self._calc_return(self.model.gamma, v))
        self.rewards.clear()

        if self.terminal:
            self.logger.info(f'Episode {episode} ended, traning: {self.model.training}, returns: {self.returns}.')
        if self.tensorboard:
            if self.model.training or shared['test_episode'] == shared['test_policy_total'] - 1:
                title = 'train/return' if self.model.training else 'test/return'
                if len(self.returns) > 1:
                    for k, v in self.returns.items():
                        self.writer.add_scalar(f'{title}/{k}', sum(v) / len(v), self.step)
                elif len(self.returns) == 1:
                    self.writer.add_scalar(title, sum(self.returns['#']) / len(self.returns['#']), self.step)
                self.returns.clear()

    def __del__(self):
        if self.terminal:
            self.logger.removeHandler(self.handler)
            self.handler.flush()
            self.handler.close()
        if self.tensorboard:
            self.writer.close()
        super().__del__()

    def _calc_return(self, gamma, rewards):
        g = 0
        for r in reversed(rewards):
            g = r + gamma * g
        return g
