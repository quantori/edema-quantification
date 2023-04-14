from typing import Any, Dict, Optional, Union, List, Sequence, Tuple, Callable, Iterable
import sys
import math
from dataclasses import dataclass

from pytorch_lightning.callbacks import TQDMProgressBar
import pytorch_lightning as pl
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import numpy as np

from models_edema import EdemaNet


# Castom progress bar for prototype updating
class PNetProgressBar(TQDMProgressBar):
    def __init__(self, process_position: int = 1):
        super().__init__(process_position=process_position)
        self._status_bar: Optional[tqdm] = None

    @property
    def status_bar(self) -> tqdm:
        if self._status_bar is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._status_bar` reference has not been set yet."
            )
        return self._status_bar

    @status_bar.setter
    def status_bar(self, bar: tqdm) -> None:
        self._status_bar = bar

    def init_status_tqdm(self) -> tqdm:
        bar = tqdm(total=0, position=1, bar_format='{desc}', file=sys.stdout)
        return bar

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        self.status_bar = self.init_status_tqdm()


def get_encoder(encoders: dict, name: str = 'squezeenet') -> nn.Module:
    try:
        return encoders[name]
    except:
        print(f'{name} encoder is not implemented')


def _compute_layer_rf_info(
    layer_filter_size, layer_stride, layer_padding, previous_layer_rf_info
) -> List[Union[int, float]]:
    # based on https://blog.mlreview.com/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    n_in = previous_layer_rf_info[0]  # receptive-field input size
    j_in = previous_layer_rf_info[1]  # receptive field jump of input layer
    r_in = previous_layer_rf_info[2]  # receptive field size of input layer
    start_in = previous_layer_rf_info[3]  # center of receptive field of input layer

    if layer_padding == 'SAME':
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if n_in % layer_stride == 0:
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
        assert (
            n_out == math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1
        )  # sanity check
        assert pad == (n_out - 1) * layer_stride - n_in + layer_filter_size  # sanity check
    elif layer_padding == 'VALID':
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert (
            n_out == math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1
        )  # sanity check
        assert pad == (n_out - 1) * layer_stride - n_in + layer_filter_size  # sanity check
    else:
        # layer_padding is an int that is the amount of padding on one side
        pad = layer_padding * 2
        # with math.floor n_out of the protoype layers has (1x1) pixels less then the
        # convulution transformations from the encoder
        # n_out = math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1
        n_out = round((n_in - layer_filter_size + pad) / layer_stride) + 1

    pL = math.floor(pad / 2)

    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1) * j_in
    start_out = start_in + ((layer_filter_size - 1) / 2 - pL) * j_in

    return [n_out, j_out, r_out, start_out]


def compute_proto_layer_rf_info(
    img_size: int, conv_info: Dict[str, int], prototype_kernel_size: int
) -> List[Union[int, float]]:
    check_dimensions(conv_info=conv_info)

    # receptive field parameters for the first layer (image itself)
    rf_info = [img_size, 1, 1, 0.5]

    for i in range(len(conv_info['kernel_sizes'])):
        filter_size = conv_info['kernel_sizes'][i]
        stride_size = conv_info['strides'][i]
        padding_size = conv_info['paddings'][i]

        rf_info = _compute_layer_rf_info(
            layer_filter_size=filter_size,
            layer_stride=stride_size,
            layer_padding=padding_size,
            previous_layer_rf_info=rf_info,
        )

    proto_layer_rf_info = _compute_layer_rf_info(
        layer_filter_size=prototype_kernel_size,
        layer_stride=1,
        layer_padding='VALID',
        previous_layer_rf_info=rf_info,
    )

    return proto_layer_rf_info


def check_dimensions(conv_info: Dict[str, int]) -> None:
    if len(conv_info['kernel_sizes']) != len(conv_info['strides']):
        raise Exception("The number of kernels has to be equla to the number of strides")
    if len(conv_info['kernel_sizes']) != len(conv_info['paddings']):
        raise Exception("The number of kernels has to be equla to the number of paddings")


def _make_layers(
    encoder: nn.Module, prototype_shape: Sequence[Union[int, float]]
) -> List[nn.Module]:
    if encoder.__class__.__name__ == "SqueezeNet":
        first_transient_layer_in_channels = (
            2 * [i for i in encoder.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        )
    else:
        first_transient_layer_in_channels = [
            i for i in encoder.modules() if isinstance(i, nn.Conv2d)
        ][-1].out_channels

    # automatic adjustment of the transient-layer channels for matching with the prototype
    # channels. The activation functions of the intermediate and last transient layers are ReLU
    # and sigmoid, respectively
    # if self.transient_layers_type == "bottleneck":
    transient_layers = []
    current_in_channels = first_transient_layer_in_channels

    while (current_in_channels > prototype_shape[1]) or (len(transient_layers) == 0):
        current_out_channels = max(prototype_shape[1], (current_in_channels // 2))
        transient_layers.append(
            nn.Conv2d(
                in_channels=current_in_channels,
                out_channels=current_out_channels,
                kernel_size=1,
            )
        )
        transient_layers.append(nn.ReLU())
        transient_layers.append(
            nn.Conv2d(
                in_channels=current_out_channels,
                out_channels=current_out_channels,
                kernel_size=1,
            )
        )

        if current_out_channels > prototype_shape[1]:
            transient_layers.append(nn.ReLU())

        else:
            assert current_out_channels == prototype_shape[1]
            transient_layers.append(nn.Sigmoid())

        current_in_channels = current_in_channels // 2

    return transient_layers


def warm(**kwargs) -> None:
    for arg in kwargs.values():
        arg.warm()


def joint(**kwargs) -> None:
    for arg in kwargs.values():
        arg.joint()


def last(**kwargs) -> None:
    for arg in kwargs.values():
        arg.last()


def print_status_bar(trainer: pl.Trainer, blocks: Dict, status: str = '') -> None:
    trainer.progress_bar_callback.status_bar.set_description_str(
        '{status}, REQUIRES GRAD: Encoder ({encoder}), Transient layers ({trans}),'
        ' Protorype layer ({prot}), Last layer ({last})'.format(
            status=status,
            encoder=get_grad_status(blocks['encoder']),
            trans=get_grad_status(blocks['transient_layers']),
            # protoype layer is inhereted from Tensor and, therefore, has the requires_grad attr
            prot=blocks['prototype_layer'].requires_grad,
            last=get_grad_status(blocks['last_layer']),
        )
    )


def get_grad_status(block: nn.Module) -> bool:
    first_param = next(block.parameters())
    if all(param.requires_grad == first_param.requires_grad for param in block.parameters()):
        return first_param.requires_grad
    else:
        raise Exception(
            f'Not all the parmaters in {block.__class__.__name__} have the same grad status'
        )


def copy_tensor_to_nparray(tensor: torch.Tensor) -> np.ndarray:
    # newer versions of PyTorch (at least 2.0.0) have numpy(force=False), where the force flag
    # substitutes tensor.detach().cpu().resolve_conj().resolve_neg().numpy()
    return np.copy(tensor.detach().cpu().numpy())
