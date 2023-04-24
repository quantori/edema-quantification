from typing import Dict, TypeVar, Generic

from torch import nn
import torch
from torchvision import transforms

T_co = TypeVar('T_co', covariant=True)


class IEncoderEdema(nn.Module, Generic[T_co]):
    """Abstract class for edema encoders."""

    def forward(self, x: T_co) -> T_co:
        """Implement this for the forward pass of the encoder."""
        raise NotImplementedError

    def conv_info(self) -> Dict[str, int]:
        """Reterns convolutional information on the encoder."""
        raise NotImplementedError

    def warm(self) -> None:
        """Sets grad policy for the warm training stage."""
        raise NotImplementedError

    def joint(self) -> None:
        """Sets grad policy for the joint training stage."""
        raise NotImplementedError

    def last(self) -> None:
        """Sets grad policy for the last training stage."""
        raise NotImplementedError


class SqueezeNet(IEncoderEdema[torch.Tensor]):
    """SqueezeNet encoder.

    The pre-trained model expects input images normalized in the same way, i.e. mini-batches of
    3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The
    images have to be loaded in to a range of [0, 1] and then normalized using
    mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    Args:
        preprocessed: flag for preprocessing an input image. Defaults to False.
        pretrained: flag for using pretrained weights. Defaults to True.
    """

    def __init__(
        self,
        preprocessed: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=pretrained, verbose=False
        )
        del self.model.classifier
        self._preprocessed = preprocessed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward implementation.

        Uses only the model.features component of SqueezeNet without model.classifier.

        Args:
            x: raw input in format (batch, channels, spatial, spatial)

        Returns:
            torch.Tensor: convolution layers after passing the SqueezNet backbone
        """
        if self._preprocessed:
            x = self.preprocess(x)
        return self.model.features(x)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return preprocess(x)

    def conv_info(self) -> Dict[str, int]:
        """Reterns info about the convolutional layers of the encoder."""
        features = {}
        features['kernel_sizes'] = []
        features['strides'] = []
        features['paddings'] = []
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
                if isinstance(module.kernel_size, tuple):
                    features['kernel_sizes'].append(module.kernel_size[0])
                else:
                    features['kernel_sizes'].append(module.kernel_size)

                if isinstance(module.stride, tuple):
                    features['strides'].append(module.stride[0])
                else:
                    features['strides'].append(module.stride)

                if isinstance(module.padding, tuple):
                    features['paddings'].append(module.padding[0])
                else:
                    features['paddings'].append(module.padding)
        return features

    def warm(self) -> None:
        self.requires_grad_(False)

    def joint(self) -> None:
        self.requires_grad_(True)

    def last(self) -> None:
        self.requires_grad_(False)


ENCODERS = {}
ENCODERS.update({'squezee_net': SqueezeNet})
