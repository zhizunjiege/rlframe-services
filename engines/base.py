from abc import ABC
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Tuple


class SimEngineBase(ABC):
    """Abstract base class for all simulation engines."""

    @abstractmethod
    def __init__(self):
        """Init engine."""
        self._state = 'uninited'

    @abstractmethod
    def __del__(self):
        """Close engine."""
        ...

    @abstractmethod
    def control(
        self,
        cmd: Literal['init', 'start', 'pause', 'step', 'resume', 'stop', 'episode', 'param'],
        params: Dict[str, Any],
    ) -> bool:
        """Control engine.

        Args:
            cmd: Control command. `episode` means ending current episode, `param` means setting simulation parameters.
            params: Control parameters.

        Returns:
            True if success.
        """
        ...

    @abstractmethod
    def monitor(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Monitor engine.

        Returns:
            Data of simulation process.
            Logs of simulation engine.
        """
        ...

    @property
    def state(self) -> Literal['uninited', 'stopped', 'running', 'suspended']:
        """Getter of state."""
        return self._state

    @state.setter
    def state(self, value: Literal['uninited', 'stopped', 'running', 'suspended']) -> None:
        """Setter of state."""
        self._state = value

    def call(self, str_data: str = '', bin_data: bytes = b'') -> Tuple[str, bytes]:
        """Any method can be called.

        Args:
            str_data: String data.
            bin_data: Binary data.

        Returns:
            String data and binary data.
        """
        return str_data, bin_data
