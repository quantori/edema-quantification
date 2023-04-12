from abc import ABC

from omegaconf import DictConfig

class Logger(ABC):
    """"Abstract base class for loggers."""
    pass


class PrototypeLogger(Logger):
    """Logger for prototype model."""
    def __init__(self, save_cfg: DictConfig):
        pass
    