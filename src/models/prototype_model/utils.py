import sys
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm


class PNetProgressBar(TQDMProgressBar):
    """Custom progress bar for displaying the progress of prototype updating."""

    def __init__(self, process_position: int = 1):
        super().__init__(process_position=process_position)
        self._status_bar: Optional[tqdm] = None

    @property
    def status_bar(self) -> tqdm:
        if self._status_bar is None:
            raise TypeError(
                f'The `{self.__class__.__name__}._status_bar` reference has not been set yet.',
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


def copy_tensor_to_nparray(tensor: torch.Tensor) -> np.ndarray:
    # Newer versions of PyTorch (at least 2.0.0) have numpy(force=False), where the force flag
    # substitutes tensor.detach().cpu().resolve_conj().resolve_neg().numpy()
    return np.copy(tensor.detach().cpu().numpy())
