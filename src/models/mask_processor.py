from typing import Tuple

import cv2
import numpy as np


class MaskProcessor:
    """MaskProcessor is a class for processing binary masks.

    It provides methods for binarization, smoothing, and artifact removal of masks.
    """

    def __init__(
        self,
        threshold_method: str = 'otsu',
        kernel_size: Tuple[int, int] = (7, 7),
    ) -> None:
        self.threshold_method = threshold_method
        self.kernel_size = kernel_size

    def binarize_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        if self.threshold_method == 'otsu':
            _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.threshold_method == 'triangle':
            _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        else:
            raise ValueError(f'Invalid threshold_method: {self.threshold_method}')

        return mask

    def smooth_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
        filled_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_DILATE, kernel)

        return filled_mask

    def remove_artifacts(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = [cv2.contourArea(cnt) for cnt in contours]
        sorted_areas = sorted(areas, reverse=True)[:2]

        biggest_contours = [cnt for cnt, area in zip(contours, areas) if area in sorted_areas]

        mask_new = np.zeros_like(mask)
        mask_new = cv2.drawContours(mask_new, biggest_contours, -1, 255, thickness=cv2.FILLED)

        return mask_new
