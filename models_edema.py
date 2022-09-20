"""Models for the edema classification project.

The description to be filled...
"""

import torch
from torch import nn
import pytorch_lightning as pl


class SqueezeNet(nn.Module):
    """SqueezeNet backbone.

    Includes the SqueezeNet architecture plus transient layers. The transient layers are required to
    concatenate the main backbone layers with the prototype layer.

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()

        self.model = torch.hub.load("pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=True)

        # self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.model(x)


class LitModel(pl.LightningModule):
    """PyTorch Lightning model class.

    A complete model is implemented (includes the backbone, prototype and fully connected layers).

    Args:
        pl (_type_): _description_
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
        # return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        pass
        # x, y = batch
        # y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        # return loss

    def configure_optimizers(self):
        pass
        # return torch.optim.Adam(self.parameters(), lr=0.02)
