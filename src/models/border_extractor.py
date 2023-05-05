import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter


class BorderExtractor:
    """Class used to extract the binary mask, delineate the region of interest and save it."""

    def __init__(
        self,
        threshold_method: str,
        threshold_val: int,
    ) -> None:
        self.threshold_method = threshold_method
        self.threshold_val = threshold_val
        assert self.threshold_method in [
            'otsu',
            'triangle',
            'manual',
        ], f'Invalid threshold_method: {self.threshold_method}'

        if threshold_method == 'manual' and not isinstance(threshold_val, int):
            raise ValueError(
                f'Manual thresholding requires a thresholding value to be set. The threshold_val is {threshold_val}',
            )

    def binarize(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        mask_bin = mask.copy()
        if self.threshold_method == 'otsu':
            threshold_val, mask_bin = cv2.threshold(
                mask,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        elif self.threshold_method == 'triangle':
            threshold_val, mask_bin = cv2.threshold(
                mask,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
            )
        elif self.threshold_method == 'manual':
            threshold_val, mask_bin = cv2.threshold(
                mask,
                self.threshold_val,
                255,
                cv2.THRESH_BINARY,
            )
        else:
            logging.warning(f'Invalid threshold: {self.threshold_val}')

        return mask_bin

    @staticmethod
    def extract_boundary(
        mask: np.ndarray,
    ) -> np.ndarray:
        _mask = Image.fromarray(mask)
        _mask = _mask.filter(ImageFilter.ModeFilter(size=7))
        _mask = np.asarray(_mask)
        mask_border = cv2.Canny(image=_mask, threshold1=100, threshold2=200)
        return mask_border

    @staticmethod
    def overlay_mask(
        image: np.ndarray,
        mask: np.ndarray,
        output_size: Tuple[int, int] = (1024, 1024),
        color: Tuple[int, int, int] = (255, 255, 0),
    ) -> np.ndarray:
        mask = cv2.resize(mask, dsize=output_size, interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, dsize=output_size, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image[mask == 255] = color

        return image
