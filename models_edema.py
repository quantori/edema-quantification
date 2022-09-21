"""Models for the edema classification project.

The description to be filled...
"""

from typing import Any, Tuple
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms


class SqueezeNet(nn.Module):
    """SqueezeNet encoder.

    The pre-trained model expects input images normalized in the same way, i.e. mini-batches of
    3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The
    images have to be loaded in to a range of [0, 1] and then normalized using
    mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    Args:
        nn (_type_): _description_

    Returns:
        torch.nn.Module: SqueezeNet.
    """

    def __init__(
        self,
        preprocessed: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=pretrained
        )

        self.preprocessed = preprocessed

        # self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):

        if self.preprocessed:
            x = self.preprocess(x)

        return self.model(x)

    def preprocess(self, x):
        """Image preprocessing function.

        To make image preprocessing model specific and modular.

        Args:
            x (Any): input image.

        Returns:
            Any: preprocessed image.
        """

        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        return preprocess(x)


class EdemaNet(pl.LightningModule):
    """PyTorch Lightning model class.

    A complete model is implemented (includes the encoder, transient, prototype and fully connected
    layers). The transient layers are required to concatenate the main encoder layers with the
    prototype layer. The encoder is the variable part of EdemaNet, which is passed as an argument.

    Args:
        encoder (nn.Module): encoder layers implemented as a distinct class.
        num_classes (int): the number of feature classes.
        prototype_shape (Tuple): the shape of the prototypes (batch, channels, H, W).
        transient_layers_type (str): the architecture of the transient layers. If == 'bottleneck',
                                     the number of channels is adjusted automatically.

    Returns:
        pl.LightningModule: EdemaNet.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        prototype_shape: Tuple,
        transient_layers_type: str = "bottleneck",
    ):
        super().__init__()

        self.transient_layers_type = transient_layers_type
        self.num_prototypes = prototype_shape[0]

        # encoder
        self.encoder = encoder

        # transient layers
        self.transient_layers = self._make_transient_layers(self.encoder)

        # prototypes layer (do not make this just a tensor, since it will not be moved
        # automatically to gpu)
        self.prototype_layer = nn.Parameter(torch.rand(prototype_shape), requires_grad=True)

        # Last fully connected layer for the classification of edema features. The bias is not used
        # in the original paper
        self.last_layer = nn.Linear(self.num_prototypes, num_classes, bias=False)

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

    def _make_transient_layers(self, encoder: torch.nn.Module) -> torch.nn.Sequential:
        """Returns transient layers.

        Args:
            encoder (torch.nn.Module): encoder architecture.

        Returns:
            torch.nn.Sequential: transient layers as the PyTorch Sequential class.
        """

        first_transient_layer_in_channels = [
            i for i in encoder.modules() if isinstance(i, nn.Conv2d)
        ][-1].out_channels

        # automatic adjustment of the transient-layer channels for matching with the prototype
        # channels. The activation functions of the intermediate and last transient layers are ReLU
        # and sigmoid, respectively
        if self.transient_layers_type == "bottleneck":
            transient_layers = []
            current_in_channels = first_transient_layer_in_channels

            while (current_in_channels > self.prototype_shape[1]) or (len(transient_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
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

                if current_out_channels > self.prototype_shape[1]:
                    transient_layers.append(nn.ReLU())

                else:
                    assert current_out_channels == self.prototype_shape[1]
                    transient_layers.append(nn.Sigmoid())

                current_in_channels = current_in_channels // 2

            transient_layers = nn.Sequential(*transient_layers)

            return transient_layers

        # determined transient layers
        else:
            transient_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=first_transient_layer_in_channels,
                    out_channels=self.prototype_shape[1],
                    kernel_size=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.prototype_shape[1],
                    out_channels=self.prototype_shape[1],
                    kernel_size=1,
                ),
                nn.Sigmoid(),
            )

            return transient_layers
