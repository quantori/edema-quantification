from typing import Any, Dict, Optional, Union, List, NamedTuple
import sys

from pytorch_lightning.callbacks import TQDMProgressBar
import pytorch_lightning as pl
from tqdm import tqdm
from torch import nn


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

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        if pl_module.current_epoch < pl_module.num_warm_epochs:
            self.status_bar.set_description_str(
                f'WARM-UP, REQUIRES GRAD: Encoder (False),' \
                    ' Transient layers (True),' \
                    ' Protorype layer (True),' \
                    ' Last layer (True)'
            )
        else:
            self.status_bar.set_description_str(
                f'JOINT, REQUIRES GRAD: Encoder (True),' \
                    ' Transient layers (True),' \
                    ' Protorype layer (True),' \
                    ' Last layer (True)' 
            )

def get_encoder(encoders: dict, name: str = 'squezeenet'):
    try:
        return encoders[name]
    except:
        print(f'{name} encoder is not implemented')

# class EdemaNetBlock(NamedTuple):
#     body: Union[nn.Module, nn.Sequential, nn.Parameter, nn.Linear]
#     requires_grad: bool


# def set_requires_grad(
#     blocks: List[NamedTuple[Union[nn.Module, nn.Sequential, nn.Parameter, nn.Linear], bool]]
# ) -> None:
#     for block in blocks:
#         if isinstance(block.body, nn.Module):
#             set_requires_grad_module(block.body, block.requires_grad)
#         else:
#             block.body.requires_grad_(block.requires_grad)


# def set_requires_grad_module(module: nn.Module, requires_grad: bool) -> None:
#     for param in module.parameters():
#         param.requires_grad_(requires_grad)
