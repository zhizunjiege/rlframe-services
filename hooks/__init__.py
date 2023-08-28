from .autosave import AutoSave
from .logging import Logging
from .training import Training

AgentHooks = {hook.__name__: hook for hook in [
    Training,
    Logging,
    AutoSave,
]}
