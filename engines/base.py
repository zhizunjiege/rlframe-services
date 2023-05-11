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
        cmd: str,
        params: Dict[str, Any],
    ) -> bool:
        """Control engine.

        Args:
            cmd: Control command.
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

    def call(self, identity: str, str_data: str = '', bin_data: bytes = b'') -> Tuple[str, str, bytes]:
        """Any method can be called.

        Args:
            identity: Identity of method.
            str_data: String data.
            bin_data: Binary data.

        Returns:
            Identity of method, string data and binary data.
        """
        return identity, '', b''
