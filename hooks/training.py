from models.base import RLModelBase

from .base import HookBase


class Training(HookBase):
    """Auto switch training mode."""

    def __init__(self, model: RLModelBase, *, test_policy_every=100, test_policy_total=5):
        """Init hook."""
        super().__init__(model)

        self.test_policy_every = test_policy_every
        self.test_policy_total = test_policy_total

        self.train_episode = 0
        self.test_episode = 0

    def before_episode(self, episode: int):
        """Before episode."""
        every = self.train_episode % self.test_policy_every == 0
        total = self.test_episode < self.test_policy_total
        if every:
            self.model.training = not total
        if not (every and total):
            self.test_episode = 0

    def after_episode(self, episode: int):
        """After episode."""
        if self.model.training:
            self.train_episode += 1
        else:
            self.test_episode += 1
