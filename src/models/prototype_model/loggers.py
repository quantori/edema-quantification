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
        return os.getcwd() + self._dir + '/' + 'real_epoch_' + str(epoch_num) + '/'

    @staticmethod
    def _make_heatmap(upsampled_act_distances: np.ndarray) -> np.ndarray:
        # Return a normalized heatmap of activations
        rescaled_act_img_j = upsampled_act_distances - np.amin(upsampled_act_distances)
        rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
        heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        return heatmap[..., ::-1]

    @staticmethod
    def _make_imagable_mask(one_img_mask: np.ndarray) -> np.ndarray:
        # Turn 2D grayscale mask into 3D
        stacked_mask = np.stack((one_img_mask,) * 3, axis=-1)
        if np.amax(stacked_mask) > 0:
            # image_mask are the weaights for multiplying the original image
            image_mask = np.where(stacked_mask == 0, 0.3, 1)  # weight 0.3
            inverted_stacked_mask = 1 - stacked_mask
            inverted_stacked_mask[:, :, 0] *= 0.7
            inverted_stacked_mask[:, :, 1] *= 0
            inverted_stacked_mask[:, :, 2] *= 0
            return inverted_stacked_mask, image_mask
        else:
            image_mask = 1 - stacked_mask
            return stacked_mask, image_mask

    def _get_overlayed_act_img(
        self, original_img_trasnposed: np.ndarray, upsampled_act_distances: np.ndarray
    ) -> np.ndarray:
        # Overlay (upsampled) activated distances on the original image
        heatmap_act = PrototypeLoggerCompNumpy._make_heatmap(upsampled_act_distances)
        overlayed_act_img = (
            self.orig_act_img_weight * original_img_trasnposed
            + self.heatmap_act_weight * heatmap_act
        )
        return overlayed_act_img

    def _get_overlayed_mask_img(
        self, original_img: np.ndarray, prototype_class: int, masks: np.ndarray
    ) -> np.ndarray:
        # Overlay masks on the original image
        one_img_mask = masks[prototype_class]
        imagable_mask, image_mask = PrototypeLoggerCompNumpy._make_imagable_mask(one_img_mask)
        overlayed_mask_img = image_mask * original_img + imagable_mask
        return overlayed_mask_img

    def _make_composition(
        self, prototype_class: int, composition_items: Mapping[str, np.ndarray]
    ) -> Image:
        # Transpose the orig image to match the heatmap dimensions (e.g., 400x400x3)
        original_img_transposed = np.transpose(composition_items['original_img'], (1, 2, 0))
        overlayed_act_img = self._get_overlayed_act_img(
            original_img_transposed, composition_items['upsampled_act_distances']
        )
        overlayed_mask_img = self._get_overlayed_mask_img(
            original_img_transposed, prototype_class, composition_items['masks']
        )
        # Make composition: original image + overlay (activated distances) + overlay (masks)
        composition_numpy = np.concatenate(
            (original_img_transposed, overlayed_act_img, overlayed_mask_img), axis=1
        )
        # Adjust the values of the composition_numpy array in 0..255 and change float32->uint8
        composition_pil = Image.fromarray((composition_numpy * 255).astype(np.uint8))
        return composition_pil

    def _save_composition(
        self,
        prototype_idx: int,
        prototype_class: int,
        epoch_num: int,
        composition_items: Mapping[str, np.ndarray],
    ) -> None:
        # Save the composition of images
        composition = self._make_composition(prototype_class, composition_items)
        composition.save(self._get_epoch_dir(epoch_num) + f'composition_proto_{prototype_idx}.png')

    def _save_rf_prototype(self, rf_proto: np.ndarray, prototype_idx: int, epoch_num: int) -> None:
        # Save the prototype receptive field
        rf_proto_t = np.transpose(rf_proto, (1, 2, 0))
        rf_image = Image.fromarray((rf_proto_t * 255).astype(np.uint8))
        rf_image.save(self._get_epoch_dir(epoch_num) + f'rf_proto_{prototype_idx}.png')

    def _save_act_roi(self, act_roi: np.ndarray, prototype_idx: int, epoch_num: int) -> None:
        # Save the highly activated ROI (highly activated region of the whole image)
        act_roi_t = np.transpose(act_roi, (1, 2, 0))
        rf_image = Image.fromarray((act_roi_t * 255).astype(np.uint8))
        rf_image.save(self._get_epoch_dir(epoch_num) + f'act_roi_{prototype_idx}.png')

    def _make_dir(self, epoch_num: int) -> str:
        target_dir = self._get_epoch_dir(epoch_num)
        if os.path.exists(target_dir):
            return target_dir
        else:
            os.makedirs(target_dir)
            return target_dir

    def save_graphics(
        self,
        prototype_idx: int,
        prototype_class: int,
        epoch_num: int,
        rf_proto: np.ndarray,
        act_roi: np.ndarray,
        **composition_items,
    ) -> None:
        self._save_composition(prototype_idx, prototype_class, epoch_num, composition_items)
        self._save_rf_prototype(rf_proto, prototype_idx, epoch_num)
        self._save_act_roi(act_roi, prototype_idx, epoch_num)

    def save_prototype_distances(
        self, distances: np.ndarray, prototype_idx: int, epoch_num: int
    ) -> None:
        # Save activated prototype distances as numpy array (the activation function of the
        # distances is log)
        directory = self._make_dir(epoch_num)
        np.save(
            os.path.join(
                directory,
                'act_distances_' + str(prototype_idx) + '.npy',
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
                prefix + '_' + str(epoch_num) + '.json',
            ),
            'w',
        )
        f.write(boxes_json)
        f.close()
