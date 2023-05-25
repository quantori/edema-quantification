import json
import logging
import os
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from src.data.utils_sly import get_box_sizes
from src.models import smp


class LungSegmenter:
    """A segmentation model used to predict the lungs from X-ray images."""

    def __init__(
        self,
        model_dir: str,
        device: str = 'auto',
    ) -> None:
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
        img: np.ndarray,
        scale_output: bool = False,
    ) -> np.ndarray:
        img_tensor = torch.unsqueeze(self.preprocess_image(img), dim=0).to(self.device)
        prob_map = self.model(img_tensor)[0, 0, :, :].cpu().detach().numpy()
        if scale_output:
            prob_map = (prob_map * 255).astype(np.uint8)
        return prob_map

    @staticmethod
    def binarize_map(
        map: np.ndarray,
        threshold_method: str = 'otsu',
    ) -> np.ndarray:
        assert threshold_method in [
            'otsu',
            'triangle',
        ], f'Invalid threshold_method: {threshold_method}'
        if threshold_method == 'otsu':
            threshold_val, mask = cv2.threshold(
                map,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        elif threshold_method == 'triangle':
            threshold_val, mask = cv2.threshold(
                map,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
            )
        else:
            raise ValueError(f'Invalid threshold_method: {threshold_method}')

        return mask

    @staticmethod
    def smooth_mask(
        mask: np.ndarray,
    ) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Perform morphological opening to smooth the edges
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Perform morphological closing to restore the object size
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

        # Perform morphological dilation to fill the holes
        filled_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_DILATE, kernel)

        return filled_mask

    @staticmethod
    def keep_lungs(
        mask: np.ndarray,
    ) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append(area)

        sorted_areas = sorted(areas, reverse=True)
        two_biggest_areas = sorted_areas[:2]

        biggest_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area in two_biggest_areas:
                biggest_contours.append(cnt)

        mask = cv2.drawContours(mask.copy(), biggest_contours, -1, (0, 255, 0), 3)

        return mask

    @staticmethod
    def compute_lung_coords(
        mask: np.ndarray,
    ) -> dict:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # TODO: add get_box_sizes for computation of lung coordinates
        for contour in contours:
            x1, y1, lung_width, lung_height = cv2.boundingRect(contour)
            x2 = x1 + lung_width
            y2 = y1 + lung_height
            lung_size = get_box_sizes(x1, y1, x2, y2)

        return lung_size


if __name__ == '__main__':
    model_name = 'DeepLabV3+'
    img_path = 'data/demo/input/10000032_50414267.png'
    img = cv2.imread(img_path)
    model = LungSegmenter(
        model_dir=f'models/lung_segmentation/{model_name}',
        device='auto',
    )
    map = model(img=img, scale_output=True)
    mask = model.binarize_map(map=map, threshold_method='otsu')
    mask = cv2.resize(mask, (1024, 1024))
    cv2.imwrite(f'data/demo/mask_{model_name}.png', mask)
