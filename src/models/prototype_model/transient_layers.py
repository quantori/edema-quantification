from typing import List, Sequence, Union

from torch import nn


class ITransientLayers(nn.Sequential):
    """Abstract class for the transient layers of the edema model."""

    def warm(self) -> None:
        """Sets grad policy for the warm training stage."""
        raise NotImplementedError

    def joint(self) -> None:
        """Sets grad policy for the joint training stage."""
        raise NotImplementedError

    def last(self) -> None:
        """Sets grad policy for the last training stage."""
        raise NotImplementedError


class TransientLayers(ITransientLayers):
    """Transient layers."""

    def __init__(
        self,
        encoder: nn.Module,
        prototype_shape: Sequence[Union[int, float]] = (9, 512, 1, 1),
    ):
        super().__init__(*_make_layers(encoder, prototype_shape))

    def warm(self) -> None:
        self.requires_grad_(True)

    def joint(self) -> None:
        self.requires_grad_(True)

    def last(self) -> None:
        self.requires_grad_(False)


def _make_layers(
    encoder: nn.Module,
    prototype_shape: Sequence[Union[int, float]],
) -> List[nn.Module]:
    if encoder.__class__.__name__ == 'SqueezeNet':
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
    transient_layers: List[nn.Module] = []
    current_in_channels = first_transient_layer_in_channels

    while (current_in_channels > prototype_shape[1]) or (len(transient_layers) == 0):
        current_out_channels = max(prototype_shape[1], (current_in_channels // 2))
        transient_layers.append(
            nn.Conv2d(
                in_channels=current_in_channels,
                out_channels=current_out_channels,
                kernel_size=1,
            ),
        )
        transient_layers.append(nn.ReLU())
        transient_layers.append(
            nn.Conv2d(
                in_channels=current_out_channels,
                out_channels=current_out_channels,
                kernel_size=1,
            ),
        )

        if current_out_channels > prototype_shape[1]:
            transient_layers.append(nn.ReLU())

        else:
            assert current_out_channels == prototype_shape[1]
            transient_layers.append(nn.Sigmoid())

        current_in_channels = current_in_channels // 2

    return transient_layers
