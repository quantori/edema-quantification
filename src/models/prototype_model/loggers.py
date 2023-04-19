import json
import os
from abc import ABC, abstractclassmethod
from typing import Union, Dict, Sequence, Mapping, Optional, TypeVar, Generic

from omegaconf import DictConfig
import numpy as np
import torch
from PIL import Image
import cv2

DIST_co = TypeVar('DIST_co', covariant=True)
BOXES_co = TypeVar('BOXES_co', covariant=True)


class PrototypeLogger(ABC, Generic[DIST_co, BOXES_co]):
    """Abstract base class for prototype loggers."""

    @abstractclassmethod
    def save_graphics(self, *args, **kwargs) -> None:
        """Called when graphical data need to be saved."""
        raise NotImplementedError

    @abstractclassmethod
    def save_prototype_distances(
        self, distances: DIST_co, prototype_idx: int, *args, **kwargs
    ) -> None:
        """Called when prototype distances need to be saved."""
        raise NotImplementedError

    @abstractclassmethod
    def save_boxes(self, boxes: BOXES_co, *args, **kwargs) -> None:
        """Called when receptive field and/or bound boxes data need to be saved."""
        raise NotImplementedError


class PrototypeLoggerComp(
    PrototypeLogger[np.ndarray, Dict[int, Dict[str, Union[int, Sequence[int]]]]]
):
    """Logger for prototypes data.

    Args:
        save_config: config with save dir(s).
    """

    def __init__(self, save_config: DictConfig) -> None:
        self._save_config = save_config

    def _get_epoch_dir(self, epoch_num: int) -> str:
        return self._save_config.dir + '/' + str(epoch_num) + '/'

    @staticmethod
    def _make_heatmap(
        original_image: np.ndarray, upsampled_act_distances: np.ndarray
    ) -> np.ndarray:
        # Return a normalized heatmap of activations
        rescaled_act_img_j = upsampled_act_distances - np.amin(upsampled_act_distances)
        rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
        heatmap = cv2.applyColorMap(
            np.uint8(255 * rescaled_act_img_j),
            cv2.COLORMAP_JET,
        )
        heatmap = np.float32(heatmap) / 255
        return heatmap[..., ::-1]

    def _make_composition(comp_weights: Mapping[str, float], **kwargs) -> Image:
        # Overlay (upsampled) activated distances on the original image
        heatmap_act = PrototypeLoggerComp._make_heatmap(
            kwargs['original_img'], kwargs['upsampled_act_distances']
        )
        overlayed_act_img = (
            comp_weights['orig_act_img'] * kwargs['original_img']
            + comp_weights['heatmap_act'] * heatmap_act
        )

        # Overlay masks on the original image
        one_img_mask = np.sum(kwargs['masks'], axis=0)
        overlayed_mask_img = (
            comp_weights['orig_mask_img'] * kwargs['original_img']
            + comp_weights['heatmap_mask'] * one_img_mask
        )

        # Make composition: original image + overlay (activated distances) + overlay (masks)
        composition_numpy = np.concatenate(
            (kwargs['original_img'], overlayed_act_img, overlayed_mask_img), axis=1
        )
        composition_pil = Image.fromarray(composition_numpy)
        return composition_pil

    def save_graphics(
        self,
        prototype_idx: int,
        orig_act_img_weight: float = 0.5,
        heatmap_act_weight: float = 0.3,
        orig_mask_img_weight: float = 0.5,
        heatmap_mask_weight: float = 0.3,
        **kwargs,
    ) -> None:
        # Save the composition of images
        composition = self._make_composition(
            {
                'orig_act_img': orig_act_img_weight,
                'heatmap_act': heatmap_act_weight,
                'orig_mask_img': orig_mask_img_weight,
                'heatmap_mask': heatmap_mask_weight,
            },
            **kwargs,
        )
        composition.save(self._get_epoch_dir + f'composition_proto_{prototype_idx}.png')

        # Save the prototype receptive field
        rf_image = Image.fromarray(kwargs['rf_proto'])
        rf_image.save(self._get_epoch_dir + f'rf_proto_{prototype_idx}.png')

        # Save the highly activated ROI (highly activated region of the whole image)
        rf_image = Image.fromarray(kwargs['act_roi'])
        rf_image.save(self._get_epoch_dir + f'act_roi_{prototype_idx}.png')

    def save_prototype_distances(
        self, distances: np.ndarray, prototype_idx: int, epoch_num: int
    ) -> None:
        # Save activated prototype distances as numpy array (the activation function of the
        # distances is log)
        np.save(
            os.path.join(
                self._get_epoch_dir(epoch_num),
                'act_distances' + str(prototype_idx) + '.npy',
            ),
            distances,
        )

    def save_boxes(
        self,
        boxes: Dict[int, Dict[str, Union[int, Sequence[int]]]],
        prefix: str,
        epoch_num: int,
    ) -> None:
        # This implementation saves in json format
        boxes_json = json.dumps(boxes)
        f = open(
            os.path.join(
                self._get_epoch_dir(epoch_num),
                prefix + str(epoch_num) + '.json',
            ),
            'w',
        )
        f.write(boxes_json)
        f.close()
