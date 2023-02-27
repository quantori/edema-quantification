from typing import Any, Tuple
from dataclasses import dataclass

from blocks import SqueezeNet


@dataclass
class EdemaNetSettings:
    encoder: Any = SqueezeNet()
    num_classes: int = 9
    prototype_shape: Tuple = (9, 512, 1, 1)
    top_k: int = 1
    epsilon: float = 1e-4
    num_warm_epochs: int = 0
    push_start: int = 0
    push_epochs: Tuple = (0, 2, 4)
    img_size: int = 400
