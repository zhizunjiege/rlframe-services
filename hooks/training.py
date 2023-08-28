from models.base import RLModelBase

from .base import HookBase, AnyDict


class Training(HookBase):
    """Auto switch training mode."""

    def __init__(self, model: RLModelBase, *, test_policy_every=100, test_policy_total=5):
        """Init hook.

        Args:
            model: RLModel instance.
            test_policy_every: Test policy every test_policy_every steps.
            test_policy_total: Test policy total test_policy_total times.
        """
        super().__init__(model)

        self.test_policy_every = test_policy_every
        self.test_policy_total = test_policy_total

        self.train_episode = 0
        self.test_episode = 0

    def before_episode(self, episode: int, shared: AnyDict):
        every = self.train_episode % self.test_policy_every == 0
        total = self.test_episode < self.test_policy_total

        self.model.training = not (every and total)

        if not total:
            self.test_episode = 0

        shared['test_policy_total'] = self.test_policy_total
        shared['test_episode'] = self.test_episode

    def after_episode(self, episode: int, shared: AnyDict):
        if self.model.training:
            self.train_episode += 1
        else:
            self.test_episode += 1
