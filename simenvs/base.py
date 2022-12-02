from abc import ABC
from abc import abstractmethod
from typing import Any, Dict, Literal, Tuple


class SimEnvBase(ABC):
    """Abstract base class for all simulation enviroments."""

    @abstractmethod
    def __init__(self, id: str, **kargs) -> None:
        """Init env.

        Args:
            id: Id of simulation enviroment.
            kargs: Any other parameters.
        """
        ...

    @abstractmethod
    def control(
        self,
        cmd: Literal['start', 'pause', 'step', 'resume', 'stop', 'done', 'param'],
        params: Dict[str, Any],
    ) -> bool:
        """Control env.

        Args:
            cmd: Control command. `done` means ending current episode, `param` means setting simulation parameters.
            params: Control parameters.

        Returns:
            supported: True if supported, False otherwise.
        """
        ...

    @abstractmethod
    def monitor(self) -> Tuple[Any, Any]:
        """Monitor env.

        Returns:
            data: Data of simulation.
            logs: Logs of simulation enviroment.
        """
        ...

    @abstractmethod
    def close(self) -> bool:
        """Close env.

        Returns:
            success: True if success, False otherwise.
        """
        ...
