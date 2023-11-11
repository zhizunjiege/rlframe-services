from datetime import datetime
import json
import os
import pickle

from models.base import RLModelBase

from .base import HookBase, AnyDict


class AutoSave(HookBase):
    """Auto save model weights, buffer and status."""

    def __init__(
        self,
        model: RLModelBase,
        *,
        per_steps=10000,
        per_episodes=100,
        save_weights: True,
        save_buffer: False,
        save_status: False,
    ):
        """Init hook.

        Args:
            model: RLModel instance.
            per_steps: Save every per_steps steps.
            per_episodes: Save every per_episodes episodes.
            save_weights: Whether to save weights.
            save_buffer: Whether to save buffer.
            save_status: Whether to save status.
        """
        super().__init__(model)

        self.per_steps = per_steps
        self.per_episodes = per_episodes

        self.save_weights = save_weights
        self.save_buffer = save_buffer
        self.save_status = save_status

        self.savedir = f'data/save/{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        self.train_steps = 0
        self.train_episodes = 0

        self.save('init')

    def after_train(self, infos: AnyDict):
        if self.model.training:
            self.train_steps += 1
            if self.per_steps is not None and self.train_steps % self.per_steps == 0:
                self.save(f'step-{self.train_steps}')

    def after_episode(self, shared: AnyDict):
        if self.model.training:
            self.train_episodes += 1
            if self.per_episodes is not None and self.train_episodes % self.per_episodes == 0:
                self.save(f'episode-{self.train_episodes}')

    def save(self, subdir: str):
        subdir = f'{self.savedir}/{subdir}'
        os.makedirs(subdir, exist_ok=True)
        if self.save_weights:
            with open(f'{subdir}/weights.pkl', 'wb') as f:
                pickle.dump(self.model.get_weights(), f)
        if self.save_buffer:
            with open(f'{subdir}/buffer.pkl', 'wb') as f:
                pickle.dump(self.model.get_buffer(), f)
        if self.save_status:
            with open(f'{subdir}/status.json', 'w') as f:
                json.dump(self.model.get_status(), f)
