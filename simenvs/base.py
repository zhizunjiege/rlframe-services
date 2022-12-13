from abc import ABC
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Tuple


class SimEnvBase(ABC):
    """Abstract base class for all simulation enviroments."""

    @abstractmethod
    def __init__(self, id: str, **kargs) -> None:
        """Init env.

        Args:
            id: Id of simulation enviroment.
            kargs: Any other parameters.
        """
        self.id = id

    @abstractmethod
    def control(
        self,
        cmd: Literal['init', 'start', 'pause', 'step', 'resume', 'stop', 'done', 'param'],
        params: Dict[str, Any],
    ) -> bool:
        """Control env.

        Args:
            cmd: Control command. `done` means ending current episode, `param` means setting simulation parameters.
            params: Control parameters.

        Returns:
            True if supported, False otherwise.
        """
        ...

    @abstractmethod
    def monitor(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Monitor env.

        Returns:
            Data of simulation.
            Logs of simulation enviroment.
        """
        ...

    @abstractmethod
    def close(self) -> bool:
        """Close env.

        Returns:
            True if success, False otherwise.
        """
        ...
