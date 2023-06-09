from typing import Dict, Generic, List, TypeVar

import torch
from torch import nn
from torchvision import transforms

T = TypeVar('T')

# Encoders for the prototype model.
ENCODERS = {}


class IEncoderEdema(nn.Module, Generic[T]):
    """Abstract class for edema encoders."""

    def forward(self, x: T) -> T:
        """Implement this for the forward pass of the encoder."""
        raise NotImplementedError

    def conv_info(self) -> Dict[str, List[int]]:
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
            'pytorch/vision:v0.10.0',
            'squeezenet1_1',
            pretrained=pretrained,
            verbose=False,
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
            ],
        )
        return preprocess(x)

    def conv_info(self) -> Dict[str, List[int]]:
        """Reterns info about the convolutional layers of the encoder."""
        features: Dict[str, List[int]] = {}
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


class ResNet50(IEncoderEdema[torch.Tensor]):
    """ResNet50 encoder.

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
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
        # Delete last layers (avgpool and fc) for attaching 'transient layaers'.
        self.model = torch.nn.Sequential(*(list(resnet.children())[:-2]))
        self._preprocessed = preprocessed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward implementation.

        Uses only the model.features component of SqueezeNet without model.classifier.

        Args:
            x: raw input in format (batch, channels, H, W)

        Returns:
            torch.Tensor: convolution layers after passing the SqueezNet backbone
        """
        if self._preprocessed:
            x = self.preprocess(x)

        return self.model(x)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
        return preprocess(x)

    def conv_info(self) -> Dict[str, List[int]]:
        """Reterns info about the convolutional layers of the encoder."""
        features: Dict[str, List[int]] = {}
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


class VGG16(IEncoderEdema[torch.Tensor]):
    """vgg16 encoder.

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
        vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=pretrained)
        # Remove last layers (classifier and avgpool) for attaching 'transient layaers'.
        self.model = torch.nn.Sequential(*(list(vgg16.children())[:-2]))
        # Remove last MaxPool2d layer.
        del self.model._modules['0']._modules['30']
        self._preprocessed = preprocessed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward implementation.

        Uses only the model.features component of SqueezeNet without model.classifier.

        Args:
            x: raw input in format (batch, channels, H, W)

        Returns:
            torch.Tensor: convolution layers after passing the SqueezNet backbone
        """
        if self._preprocessed:
            x = self.preprocess(x)

        return self.model(x)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
        return preprocess(x)

    def conv_info(self) -> Dict[str, List[int]]:
        """Reterns info about the convolutional layers of the encoder."""
        features: Dict[str, List[int]] = {}
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


ENCODERS.update({'squezee_net': SqueezeNet})
ENCODERS.update({'resnet50': ResNet50})
ENCODERS.update({'vgg16': VGG16})


if __name__ == '__main__':
    net = VGG16()
    print(net.conv_info())
