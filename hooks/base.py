from abc import ABC
from typing import Any, Dict

from models.base import RLModelBase

AnyDict = Dict[str, Any]


class HookBase(ABC):
    """Abstract base class for all hooks."""

    @classmethod
    @property
    def name(self) -> str:
        """Return name of this hook."""
        return self.__name__

    def __init__(self, model: RLModelBase):
        """Init hook."""
        self.model = model

    def before_episode(self, episode: int):
        """Before episode."""
        ...

    def before_react(self, step: int):
        """Before react."""
        ...

    def after_react(self, step: int, siargs: AnyDict, oaargs: AnyDict, caches: AnyDict):
        """After react."""
        ...

    def react2train(self, step: int, rewargs: AnyDict, caches: AnyDict):
        """Between react and train."""
        ...

    def before_train(self, step: int):
        """Before train."""
        ...

    def after_train(self, step: int, infos: AnyDict):
        """After train."""
        ...

    def after_episode(self, episode: int):
        """After episode."""
        ...

    def __del__(self):
        """Close hook."""
        self.model = None
