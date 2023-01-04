from abc import ABC
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Tuple


class SimEngineBase(ABC):
    """Abstract base class for all simulation engines."""

    @abstractmethod
    def __init__(self) -> None:
        """Init engine."""
        self._state = 'uninited'

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

    @abstractmethod
    def close(self) -> bool:
        """Close engine.

        Returns:
            True if success.
        """
        ...

    @property
    def state(self) -> Literal['uninited', 'stopped', 'running', 'suspended']:
        """Getter of state."""
        return self._state
