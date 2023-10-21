from abc import ABC
from typing import Any, Dict

from models.base import RLModelBase

AnyDict = Dict[str, Any]

LOGGER_NAME = 'hook'


class HookBase(ABC):
    """Abstract base class for all hooks."""

    def __init__(self, model: RLModelBase):
        """Init hook.

        Args:
            model: RLModel instance.
        """
        self.model = model

    def before_episode(self, shared: AnyDict):
        """Before episode.

        Args:
            shared: Shared data between hooks.
        """
        ...

    def before_react(self, siargs: AnyDict):
        """Before react.

        Args:
            siargs: Arguments of state to input function.
        """
        ...

    def after_react(self, oaargs: AnyDict):
        """After react.

        Args:
            oaargs: Arguments of output to action function.
        """
        ...

    def react_train(self, rewargs: AnyDict):
        """Between react and train.

        Args:
            rewargs: Arguments of reward function.
        """
        ...

    def before_train(self, data: AnyDict):
        """Before train.

        Args:
            data: MDP data to store.
        """
        ...

    def after_train(self, infos: AnyDict):
        """After train.

        Args:
            infos: Returned informations of train function.
        """
        ...

    def after_episode(self, shared: AnyDict):
        """After episode.

        Args:
            shared: Shared data between hooks.
        """
        ...

    def __del__(self):
        """Close hook."""
        self.model = None
