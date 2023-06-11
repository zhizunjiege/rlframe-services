from .autosave import AutoSave
from .tensorboard import Tensorboard
from .training import Training

AgentHooks = {hook.name: hook for hook in [
    Training,
    Tensorboard,
    AutoSave,
]}
