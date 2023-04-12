import json
import os
from abc import ABC, abstractclassmethod
from typing import Union, Dict, Sequence

from omegaconf import DictConfig
import numpy as np
import torch


class PrototypeLogger(ABC):
    """Abstract base class for loggers.

    Subclass this class and override any of the relevant save methods.
    """

    @abstractclassmethod
    def save_graphics(self) -> None:
        """Called when graphical data need to be saved"""

    @abstractclassmethod
    def save_prototype_representations(self) -> None:
        """Called when embeddings of the prototypes need to be saved"""

    @abstractclassmethod
    def save_rf_boxes(self) -> None:
        """Called when receptive field and bounding boxes need to be saved"""

    @abstractclassmethod
    def save_bound_boxes(self) -> None:
        """Called when receptive field and bounding boxes need to be saved"""


class PrototypeLogger1(PrototypeLogger):
    """Logger for prototype model.

    Args:
        save_config: config with save path(s).
    """

    def __init__(self, save_config: DictConfig) -> None:
        self._save_config = save_config
        self._with_epoch = True

    @property
    def with_epoch(self) -> bool:
        return self._with_epoch

    def _mkdir_epoch(self, epoch_num: int) -> str:
        return self._save_config.path + '/' + str(epoch_num)

    def save_graphics(self, img: np.ndarray) -> None:
        pass

    def save_prototype_representations(self, array: np.ndarray):
        pass

    def save_rf_boxes(
        self,
        boxes: Dict[int, Dict[str, Union[int, Sequence[int]]]],
        epoch_num: int,
    ) -> None:
        # This implementation saves in json format
        boxes_json = json.dumps(boxes)
        f = open(
            os.path.join(
                self._mkdir_epoch(epoch_num),
                'rf' + str(epoch_num) + '.json',
            ),
            'w',
        )
        f.write(boxes_json)
        f.close()

    def save_bound_boxes(
        self,
        boxes: Dict[int, Dict[str, Union[int, Sequence[int]]]],
        epoch_num: int,
    ) -> None:
        # This implementation saves in json format
        boxes_json = json.dumps(boxes)
        f = open(
            os.path.join(
                self._mkdir_epoch(epoch_num),
                'bound' + str(epoch_num) + '.json',
            ),
            'w',
        )
        f.write(boxes_json)
        f.close()
