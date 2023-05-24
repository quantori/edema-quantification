import json
import logging
import os
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from src.models import smp


class LungSegmenter:
    """Class used to predict lungs on X-ray images."""

    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.5,
        device: str = 'auto',
        raw_output: bool = False,
    ) -> None:
        assert (
            0 <= threshold <= 1
        ), f'Threshold should be in the range [0,1], while it is {threshold}'

        # Model settings
        f = open(os.path.join(model_dir, 'config.json'))
        _model_params = json.load(f)
        model_params = _model_params['parameters']
        _input_size = model_params['input_size']
        self.input_size = (
            (_input_size, _input_size) if isinstance(_input_size, int) else tuple(_input_size)
        )
        self.model_name = model_params['model_name']
        self.encoder_name = model_params['encoder_name']
        self.encoder_weights = model_params['encoder_weights']
        self.input_channels = model_params['input_channels']
        self.num_classes = model_params['num_classes']
        self.activation = model_params['activation']
        self.threshold = threshold
        self.raw_output = raw_output
        self.preprocessing = smp.encoders.get_preprocessing_params(
            encoder_name=self.encoder_name,
            pretrained=self.encoder_weights,
        )
        self.preprocess_image = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    size=self.input_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.Normalize(
                    mean=self.preprocessing['mean'],
                    std=self.preprocessing['std'],
                ),
            ],
        )

        # Build model
        _model = self.build_model()

        if device == 'auto':
            selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cpu':
            selected_device = 'cpu'
        elif device == 'gpu':
            selected_device = 'cuda'
        else:
            raise ValueError(f'Unsupported device type: {device}')
        self.device = selected_device
        self.model = _model.to(self.device)
        self.model.load_state_dict(
            torch.load(
                f=os.path.join(model_dir, 'weights.pth'),
                map_location=self.device,
            ),
        )
        self.model.eval()

        # Log model parameters
        logging.info('Model.....................:')
        logging.info(f'Model dir.................: {model_dir}')
        logging.info(f'Model name................: {self.model_name}')
        logging.info(f'Input size................: {self.input_size}')
        logging.info(f'Threshold.................: {self.threshold}')
        logging.info(f'Raw output................: {self.raw_output}')
        logging.info(f'Device....................: {self.device.upper()}')

    def build_model(self) -> Any:
        if self.model_name == 'Unet':
            model = smp.Unet(
                encoder_name=self.encoder_name,
                encoder_weights=None,
                in_channels=self.input_channels,
                classes=self.num_classes,
                activation=self.activation,
            )
        elif self.model_name == 'Unet++':
            model = smp.UnetPlusPlus(
                encoder_name=self.encoder_name,
                encoder_weights=None,
                in_channels=self.input_channels,
                classes=self.num_classes,
                activation=self.activation,
            )
        elif self.model_name == 'DeepLabV3':
            model = smp.DeepLabV3(
                encoder_name=self.encoder_name,
                encoder_weights=None,
                in_channels=self.input_channels,
                classes=self.num_classes,
                activation=self.activation,
            )
        elif self.model_name == 'DeepLabV3+':
            model = smp.DeepLabV3Plus(
                encoder_name=self.encoder_name,
                encoder_weights=None,
                in_channels=self.input_channels,
                classes=self.num_classes,
                activation=self.activation,
            )
        elif self.model_name == 'FPN':
            model = smp.FPN(
                encoder_name=self.encoder_name,
                encoder_weights=None,
                in_channels=self.input_channels,
                classes=self.num_classes,
                activation=self.activation,
            )
        elif self.model_name == 'Linknet':
            model = smp.Linknet(
                encoder_name=self.encoder_name,
                encoder_weights=None,
                in_channels=self.input_channels,
                classes=self.num_classes,
                activation=self.activation,
            )
        elif self.model_name == 'PSPNet':
            model = smp.PSPNet(
                encoder_name=self.encoder_name,
                encoder_weights=None,
                in_channels=self.input_channels,
                classes=self.num_classes,
                activation=self.activation,
            )
        elif self.model_name == 'PAN':
            model = smp.PAN(
                encoder_name=self.encoder_name,
                encoder_weights=None,
                in_channels=self.input_channels,
                classes=self.num_classes,
                activation=self.activation,
            )
        elif self.model_name == 'MAnet':
            model = smp.MAnet(
                encoder_name=self.encoder_name,
                encoder_weights=None,
                in_channels=self.input_channels,
                classes=self.num_classes,
                activation=self.activation,
            )
        else:
            raise ValueError('Unknown model name: {:s}'.format(self.model_name))

        return model

    def __call__(
        self,
        img_path: str,
    ) -> np.ndarray:
        img = cv2.imread(img_path)
        img_tensor = torch.unsqueeze(self.preprocess_image(img), dim=0).to(self.device)
        mask = self.model(img_tensor)[0, 0, :, :].cpu().detach().numpy()
        if self.raw_output:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask >= self.threshold
            mask = 255 * mask.astype(np.uint8)
        return mask


if __name__ == '__main__':
    model_name = 'DeepLabV3+'
    img_path = 'data/demo/input/10000032_50414267.png'
    model = LungSegmenter(
        model_dir=f'models/lung_segmentation/{model_name}',
        threshold=0.50,
        device='auto',
        raw_output=True,
    )
    mask = model(img_path)
    mask = cv2.resize(mask, (1024, 1024))
    cv2.imwrite(f'data/demo/mask_{model_name}.png', mask)
