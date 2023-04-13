import json
import os
from abc import ABC, abstractclassmethod
from typing import Union, Dict, Sequence, Optional

from omegaconf import DictConfig
import numpy as np
import torch


class PrototypeLogger(ABC):
    """Abstract base class for prototype loggers."""

    @abstractclassmethod
    def save_graphics(self, *args, **kwargs) -> None:
        """Called when graphical data need to be saved."""
        raise NotImplementedError

    @abstractclassmethod
    def save_prototype_embeddings(self, *args, **kwargs) -> None:
        """Called when prototype embeddings need to be saved."""
        raise NotImplementedError

    @abstractclassmethod
    def save_boxes(self, *args, **kwargs) -> None:
        """Called when receptive field and/or bound boxes data need to be saved."""
        raise NotImplementedError


class PrototypeLogger1(PrototypeLogger):
    """Logger for prototypes data.

    Args:
        save_config: config with save dir(s).
    """

    def __init__(self, save_config: DictConfig) -> None:
        self._save_config = save_config

    def _mkdir_epoch(self, epoch_num: int) -> str:
        return self._save_config.dir + '/' + str(epoch_num)

    def save_graphics(self, img: np.ndarray) -> None:
        pass

    def save_prototype_representations(self, array: np.ndarray) -> None:
        pass

    def save_boxes(
        self,
        boxes: Dict[int, Dict[str, Union[int, Sequence[int]]]],
        prefix: str,
        epoch_num: int,
    ) -> None:
        # This implementation saves in json format
        boxes_json = json.dumps(boxes)
        f = open(
            os.path.join(
                self._mkdir_epoch(epoch_num),
                prefix + str(epoch_num) + '.json',
            ),
            'w',
        )
        f.write(boxes_json)
        f.close()
