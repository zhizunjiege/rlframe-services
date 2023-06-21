from .autosave import AutoSave
from .logging import Logging
from .training import Training

AgentHooks = {hook.name: hook for hook in [
    Training,
    Logging,
    AutoSave,
]}
