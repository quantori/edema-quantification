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


class IPrototypeLogger(ABC, Generic[DIST_co, BOXES_co]):
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


class PrototypeLoggerCompNumpy(
    IPrototypeLogger[np.ndarray, Dict[int, Dict[str, Union[int, Sequence[int]]]]]
):
    """Logger for prototypes data.

    Args:
        save_config: config with save dir(s).
    """

    def __init__(
        self,
        logger_config: DictConfig,
    ) -> None:
        self._dir = logger_config.dir
        self.orig_act_img_weight = logger_config.orig_act_img_weight
        self.heatmap_act_weight = logger_config.heatmap_act_weight
        self.orig_mask_img_weight = logger_config.orig_mask_img_weight
        self.heatmap_mask_weight = logger_config.heatmap_mask_weight

    def _get_epoch_dir(self, epoch_num: int) -> str:
        return self._dir + '/' + str(epoch_num) + '/'

    @staticmethod
    def _make_heatmap(
        original_image: np.ndarray, upsampled_act_distances: np.ndarray
    ) -> np.ndarray:
        # Return a normalized heatmap of activations
        rescaled_act_img_j = upsampled_act_distances - np.amin(upsampled_act_distances)
        rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
        heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        return heatmap[..., ::-1]

    def _get_overlayed_act_img(
        self, original_img: np.ndarray, upsampled_act_distances: np.ndarray
    ) -> np.ndarray:
        # Overlay (upsampled) activated distances on the original image
        heatmap_act = PrototypeLoggerCompNumpy._make_heatmap(original_img, upsampled_act_distances)
        overlayed_act_img = (
            self.orig_act_img_weight * original_img + self.heatmap_act_weight * heatmap_act
        )
        return overlayed_act_img

    def _get_overlayed_mask_img(self, original_img: np.ndarray, masks: np.ndarray) -> np.ndarray:
        # Overlay masks on the original image
        one_img_mask = np.sum(masks, axis=0)
        overlayed_mask_img = (
            self.orig_mask_img_weight * original_img + self.heatmap_mask_weight * one_img_mask
        )
        return overlayed_mask_img

    def _make_composition(self, **composition_items) -> Image:
        overlayed_act_img = self._get_overlayed_act_img(
            composition_items['original_img'], composition_items['upsampled_act_distances']
        )
        overlayed_mask_img = self._get_overlayed_mask_img(
            composition_items['original_img'], composition_items['masks']
        )
        # Make composition: original image + overlay (activated distances) + overlay (masks)
        composition_numpy = np.concatenate(
            (composition_items['original_img'], overlayed_act_img, overlayed_mask_img), axis=1
        )
        composition_pil = Image.fromarray(composition_numpy)
        return composition_pil

    def _save_composition(self, prototype_idx: int, epoch_num: int, **composition_items) -> None:
        # Save the composition of images
        composition = self._make_composition(**composition_items)
        composition.save(self._get_epoch_dir(epoch_num) + f'composition_proto_{prototype_idx}.png')

    def _save_rf_prototype(self, rf_proto: np.ndarray, prototype_idx: int, epoch_num: int) -> None:
        # Save the prototype receptive field
        rf_image = Image.fromarray(rf_proto)
        rf_image.save(self._get_epoch_dir(epoch_num) + f'rf_proto_{prototype_idx}.png')

    def _save_act_roi(self, act_roi: np.ndarray, prototype_idx: int, epoch_num: int) -> None:
        # Save the highly activated ROI (highly activated region of the whole image)
        rf_image = Image.fromarray(act_roi)
        rf_image.save(self._get_epoch_dir(epoch_num) + f'act_roi_{prototype_idx}.png')

    def save_graphics(
        self,
        prototype_idx: int,
        epoch_num: int,
        rf_proto: np.ndarray,
        act_roi: np.ndarray,
        **composition_items,
    ) -> None:
        self._save_composition(prototype_idx, epoch_num, **composition_items)
        self._save_rf_prototype(rf_proto, prototype_idx, epoch_num)
        self._save_act_roi(act_roi, prototype_idx, epoch_num)

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
