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

        self.episode = 0
        self.global_react_steps, self.global_train_steps = 0, 0
        self.local_react_steps, self.local_train_steps = 0, 0

    def before_episode(self, shared: AnyDict):
        self.local_react_steps, self.local_train_steps = 0, 0
        if self.terminal:
            self.logger.info(f'Episode {self.episode} start...training: {self.model.training}.')

    def before_react(self, siargs: AnyDict):
        if self.terminal:
            self.logger.debug(f'React step {self.local_react_steps}/{self.global_react_steps}...')
            self.logger.debug(f'\tstates: {siargs["states"]}.')
            self.logger.debug(f'\tinputs: {siargs["inputs"]}.')

    def after_react(self, oaargs: AnyDict):
        if self.terminal:
            self.logger.debug(f'\toutputs: {oaargs["outputs"]}.')
            self.logger.debug(f'\tactions: {oaargs["actions"]}.')

        self.global_react_steps += 1
        self.local_react_steps += 1

    def react_train(self, rewargs: AnyDict):
        terminated = rewargs['terminated']
        truncated = rewargs['truncated']
        reward = rewargs['reward']

        if isinstance(reward, dict):
            for k, v in reward.items():
                self.rewards.setdefault(k, []).append(float(v))
        else:
            self.rewards.setdefault('#', []).append(float(reward))

        if self.terminal:
            self.logger.debug(f'Between react & train...terminated: {terminated}, truncated: {truncated}, reward: {reward}.')

    def before_train(self, data: AnyDict):
        if self.terminal:
            self.logger.debug(f'Train step {self.local_train_steps}/{self.global_train_steps}...')
            self.logger.debug(f'\tnext_states: {data["next_states"]}.')
            self.logger.debug(f'\tnext_inputs: {data["next_inputs"]}.')

    def after_train(self, infos: AnyDict):
        if self.terminal:
            self.logger.debug(f'\tinfos: {infos}.')

        if self.tensorboard:
            for k, v in infos.items():
                if isinstance(v, Iterable):
                    if len(v) > 0:
                        self.writer.add_scalar(f'train/{k}', float(sum(v) / len(v)), self.global_train_steps)
                else:
                    self.writer.add_scalar(f'train/{k}', float(v), self.global_train_steps)

        self.global_train_steps += 1
        self.local_train_steps += 1

    def after_episode(self, shared: AnyDict):
        if self.model.training:
            self.returns.clear()

        for k, v in self.rewards.items():
            self.returns.setdefault(k, []).append(self._calc_return(self.model.gamma, v))
        self.rewards.clear()

        if self.terminal:
            self.logger.info(f'Episode {self.episode} end after {self.local_react_steps} steps...returns: {self.returns}.')

        if self.tensorboard:
            if self.model.training or shared['test_episode'] == shared['test_policy_total'] - 1:
                title = 'train/return' if self.model.training else 'test/return'
                if len(self.returns) > 1:
                    for k, v in self.returns.items():
                        self.writer.add_scalar(f'{title}/{k}', sum(v) / len(v), self.global_train_steps)
                elif len(self.returns) == 1:
                    self.writer.add_scalar(title, sum(self.returns['#']) / len(self.returns['#']), self.global_train_steps)
                self.returns.clear()

        self.episode += 1

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
