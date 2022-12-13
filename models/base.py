from abc import ABC
from abc import abstractmethod
from typing import Any, Dict


class RLModelBase(ABC):
    """Abstract base class for all reinforcement learning models."""

    @abstractmethod
    def __init__(self, training: bool) -> None:
        """Init model.

        Args:
            training: Whether the model is in training mode.
        """
        self.__training = training

    @abstractmethod
    def react(self, states: Any) -> Any:
        """Get action from model.

        Args:
            states: States of enviroment.

        Returns:
            Action.
        """
        ...

    @abstractmethod
    def store(self, states: Any, actions: Any, next_states: Any, reward: float, done: bool) -> Any:
        """Store experience replay data.

        Args:
            states: States of enviroment.

            actions: Actions of model.

            next_states: Next states of enviroment.

            reward: Reward.

            done: Indicating whether terminated or not.
        """
        ...

    @abstractmethod
    def train(self) -> Any:
        """Train model."""
        ...

    @abstractmethod
    def get_weights(self) -> Any:
        """Get weights of neural networks.

        Returns:
            Weights of networks.
        """
        ...

    @abstractmethod
    def set_weights(self, weights: Any) -> Any:
        """Set weights of neural networks.

        Args:
            weights: Weights of networks.
        """
        ...

    @abstractmethod
    def get_buffer(self) -> Any:
        """Get buffer of experience replay.

        Returns:
            Buffer of replay.
        """
        ...

    @abstractmethod
    def set_buffer(self, buffer: Any) -> Any:
        """Set buffer of experience replay.

        Args:
            buffer: Buffer of replay.
        """
        ...

    def get_status(self) -> Dict[str, Any]:
        """Get intermediate status of model.

        Returns:
            Intermediate status.
        """
        status = {}
        for key in self.__dict__:
            if key[0] == '_' and key.find('__') == -1:
                name = key[1:]
                status[name] = self.__dict__[key]
        return status

    def set_status(self, status: Dict[str, Any]) -> None:
        """Set intermediate status of model.

        Args:
            status: Intermediate status.

        Raises:
            KeyError: When key of status is invalid.
        """
        for name in status:
            key = f'_{name}'
            if key in self.__dict__:
                setattr(self, key, status[name])
            else:
                raise KeyError(f'Key: `{name}` is invalid.')

    @property
    def training(self) -> bool:
        """Getter of training."""
        return self.__training

    @training.setter
    def training(self, value: bool):
        """Setter of training."""
        self.__training = value
