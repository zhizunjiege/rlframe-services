from datetime import datetime
import os
import pickle

from models.base import RLModelBase

from .base import HookBase, AnyDict


class AutoSave(HookBase):
    """Auto save model weights."""

    def __init__(self, model: RLModelBase, *, per_steps=10000, per_episodes=100):
        """Init hook."""
        super().__init__(model)

        self.per_steps = per_steps
        self.per_episodes = per_episodes

        self.savedir = f'data/save/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        os.makedirs(self.savedir, exist_ok=True)

        self.train_steps = 0
        self.train_episodes = 0

        self._save_weights(f'{self.savedir}/init.pkl')

    def after_train(self, step: int, infos: AnyDict):
        """After train."""
        if self.model.training:
            self.train_steps += 1
            if self.per_steps is not None and self.train_steps % self.per_steps == 0:
                self._save_weights(f'{self.savedir}/step-{self.train_steps}.pkl')

    def after_episode(self, episode: int):
        """After episode."""
        if self.model.training:
            self.train_episodes += 1
            if self.per_episodes is not None and self.train_episodes % self.per_episodes == 0:
                self._save_weights(f'{self.savedir}/episode-{self.train_episodes}.pkl')

    def _save_weights(self, path: str):
        """Save weights."""
        with open(path, 'wb') as f:
            pickle.dump(self.model.get_weights(), f)
