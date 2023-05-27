from typing import Tuple

import cv2
import numpy as np


class MaskProcessor:
    """MaskProcessor is a class for processing binary masks.

    It provides methods for binarization, smoothing, and artifact removal of masks.
    """

    @staticmethod
    def binarize_image(
        image: np.ndarray,
        threshold_method: str = 'otsu',
    ) -> np.ndarray:
        assert threshold_method in [
            'otsu',
            'triangle',
        ], f'Invalid threshold_method: {threshold_method}'

        if threshold_method == 'otsu':
            _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_method == 'triangle':
            _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        return mask

    @staticmethod
    def smooth_mask(
        mask: np.ndarray,
        kernel_size: Tuple[int, int] = (7, 7),
    ) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
        filled_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_DILATE, kernel)

        return filled_mask

    @staticmethod
    def remove_artifacts(
        mask: np.ndarray,
    ) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = [cv2.contourArea(cnt) for cnt in contours]
        sorted_areas = sorted(areas, reverse=True)[:2]

        biggest_contours = [cnt for cnt, area in zip(contours, areas) if area in sorted_areas]

        new_mask = np.zeros_like(mask)
        new_mask = cv2.drawContours(new_mask, biggest_contours, -1, 255, thickness=cv2.FILLED)

        return new_mask