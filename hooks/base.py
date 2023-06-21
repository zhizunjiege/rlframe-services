from abc import ABC
from typing import Any, Dict

from models.base import RLModelBase

AnyDict = Dict[str, Any]

LOGGER_NAME = 'hook'


class HookBase(ABC):
    """Abstract base class for all hooks."""

    @classmethod
    @property
    def name(self) -> str:
        """Return name of this hook."""
        return self.__name__

    def __init__(self, model: RLModelBase):
        """Init hook.

        Args:
            model: RLModel instance.
        """
        self.model = model

    def before_episode(self, episode: int):
        """Before episode.

        Args:
            episode: Current episode.
        """
        ...

    def before_react(self, step: int):
        """Before react.

        Args:
            step: Current react step.
        """
        ...

    def after_react(self, step: int, siargs: AnyDict, oaargs: AnyDict, caches: AnyDict):
        """After react.

        Args:
            step: Current react step.
            siargs: Arguments of state to input function.
            oaargs: Arguments of output to action function.
            caches: Caches.
        """
        ...

    def react2train(self, rewargs: AnyDict, caches: AnyDict):
        """Between react and train.

        Args:
            rewargs: Arguments of reward function.
            caches: Caches.
        """
        ...

    def before_train(self, step: int):
        """Before train.

        Args:
            step: Current train step.
        """
        ...

    def after_train(self, step: int, infos: AnyDict):
        """After train.

        Args:
            step: Current train step.
            infos: Returned informations of train function.
        """
        ...

    def after_episode(self, episode: int):
        """After episode.

        Args:
            episode: Current episode.
        """
        ...

    def __del__(self):
        """Close hook."""
        self.model = None
