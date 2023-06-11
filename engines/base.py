from abc import ABC
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Tuple

AnyDict = Dict[str, Any]
CommandType = Literal['init', 'start', 'pause', 'step', 'resume', 'stop', 'episode', 'param']
EngineState = Literal['uninited', 'stopped', 'running', 'suspended']


class SimEngineBase(ABC):
    """Abstract base class for all simulation engines."""

    @classmethod
    @property
    def name(self) -> str:
        """Return name of this engine."""
        return self.__name__

    def __init__(self):
        """Init engine."""
        self._state = 'uninited'

    @abstractmethod
    def control(
        self,
        type: CommandType,
        params: AnyDict,
    ) -> bool:
        """Control engine.

        Args:
            type: Command type.
            params: Command params.

        Returns:
            True if success.
        """
        ...

    @abstractmethod
    def monitor(self) -> Tuple[List[AnyDict], List[str]]:
        """Monitor engine.

        Returns:
            Data of simulation process.
            Logs of simulation engine.
        """
        ...

    @property
    def state(self) -> EngineState:
        """Getter of state."""
        return self._state

    @state.setter
    def state(self, value: EngineState) -> None:
        """Setter of state."""
        self._state = value

    def call(self, name: str, dstr='', dbin=b'') -> Tuple[str, str, bytes]:
        """Any method can be called.

        Args:
            name: Name of method.
            dstr: String data.
            dbin: Binary data.

        Returns:
            Name of method, string data and binary data.
        """
        return name, '', b''

    def __del__(self):
        """Close engine."""
        ...
