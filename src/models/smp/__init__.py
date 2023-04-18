from typing import Optional

import torch

from . import encoders, losses, utils
from .__version__ import __version__
from .deeplabv3 import DeepLabV3, DeepLabV3Plus
from .fpn import FPN
from .linknet import Linknet
from .manet import MAnet
from .pan import PAN
from .pspnet import PSPNet
from .unet import Unet
from .unetplusplus import UnetPlusPlus


def create_model(
    arch: str,
    encoder_name: str = 'resnet34',
    encoder_weights: Optional[str] = 'imagenet',
    in_channels: int = 3,
    classes: int = 1,
    **kwargs,
) -> torch.nn.Module:
    """Models wrapper. Allows to create any model just with parametes"""

    archs = [Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN]
    archs_dict = {a.__name__.lower(): a for a in archs}
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError(
            'Wrong architecture type `{}`. Available options are: {}'.format(
                arch,
                list(archs_dict.keys()),
            ),
        )
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )
