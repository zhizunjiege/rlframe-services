from .cqsim import CQSIM

SimEngines = {engine.name: engine for engine in [
    CQSIM,
]}
