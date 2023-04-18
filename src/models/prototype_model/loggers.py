import json
import os
from abc import ABC, abstractclassmethod
from typing import Union, Dict, Sequence, Optional, TypeVar, Generic

from omegaconf import DictConfig
import numpy as np
import torch
from PIL import Image

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
        return self._save_config.dir + '/' + str(epoch_num)

    def _make_composition(
        original_img: np.ndarray, upsampled_act_distances: np.ndarray, masks: np.ndarray
    ) -> Image:
        pass

    def save_graphics(
        self, rf_of_prototype: np.ndarray, highly_act_roi: np.ndarray, **kwargs
    ) -> None:
        composition = self._make_composition(
            kwargs['original_img'], kwargs['upsampled_act_distances'], kwargs['masks']
        )

        # if prototype_img_filename_prefix is not None:

    #                     # save the whole image containing the prototype as png
    #                     plt.imsave(
    #                         os.path.join(
    #                             dir_for_saving_prototypes,
    #                             prototype_img_filename_prefix + '-original' + str(j) + '.png',
    #                         ),
    #                         original_img_j,
    #                         cmap='gray',
    #                     )

    #                     # overlay (upsampled) activation on original image and save the result
    #                     # normalize the image
    #                     rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
    #                     rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
    #                     heatmap = cv2.applyColorMap(
    #                         np.uint8(255 * rescaled_act_img_j),
    #                         cv2.COLORMAP_JET,
    #                     )
    #                     heatmap = np.float32(heatmap) / 255
    #                     heatmap = heatmap[..., ::-1]
    #                     overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
    #                     plt.imsave(
    #                         os.path.join(
    #                             dir_for_saving_prototypes,
    #                             prototype_img_filename_prefix
    #                             + '-original_with_self_act'
    #                             + str(j)
    #                             + '.png',
    #                         ),
    #                         overlayed_original_img_j,
    #                         vmin=0.0,
    #                         vmax=1.0,
    #                     )

    #                     # if different from the original (whole) image, save the prototype receptive
    #                     # field as png
    #                     if (
    #                         rf_img_j.shape[0] != original_img_size
    #                         or rf_img_j.shape[1] != original_img_size
    #                     ):
    #                         plt.imsave(
    #                             os.path.join(
    #                                 dir_for_saving_prototypes,
    #                                 prototype_img_filename_prefix
    #                                 + '-receptive_field'
    #                                 + str(j)
    #                                 + '.png',
    #                             ),
    #                             rf_img_j,
    #                             vmin=0.0,
    #                             vmax=1.0,
    #                         )
    #                         overlayed_rf_img_j = overlayed_original_img_j[
    #                             rf_prototype_j[1] : rf_prototype_j[2],
    #                             rf_prototype_j[3] : rf_prototype_j[4],
    #                         ]
    #                         plt.imsave(
    #                             os.path.join(
    #                                 dir_for_saving_prototypes,
    #                                 prototype_img_filename_prefix
    #                                 + '-receptive_field_with_self_act'
    #                                 + str(j)
    #                                 + '.png',
    #                             ),
    #                             overlayed_rf_img_j,
    #                             vmin=0.0,
    #                             vmax=1.0,
    #                         )

    #                     # save the highly activated ROI (highly activated region of the whole image)
    #                     plt.imsave(
    #                         os.path.join(
    #                             dir_for_saving_prototypes,
    #                             prototype_img_filename_prefix + str(j) + '.png',
    #                         ),
    #                         proto_img_j,
    #                         vmin=0.0,
    #                         vmax=1.0,
    #                     )

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
