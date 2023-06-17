import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd

from src.data.utils_sly import FEATURE_MAP, get_box_sizes
from src.models.edema_classifier import EdemaClassifier
from src.models.feature_detector import FeatureDetector
from src.models.lung_segmenter import LungSegmenter
from src.models.map_fuser import MapFuser
from src.models.mask_processor import MaskProcessor
from src.models.non_max_suppressor import NonMaxSuppressor


class EdemaNet:
    """A network dedicated to processing X-ray images and predicting the stage of edema."""

    IMG_NAME = 'img.png'
    IMG_CROP_NAME = 'img_crop.png'
    MASK_NAME = 'mask.png'
    MASK_CROP_NAME = 'mask_crop.png'
    MAP_NAME = 'map.png'
    MAP_PREFIX = 'map'
    METADATA_NAME = 'metadata.xlsx'

    def __init__(
        self,
        lung_segmenters: List[LungSegmenter],
        feature_detectors: List[FeatureDetector],
        map_fuser: MapFuser,
        mask_processor: MaskProcessor,
        non_max_suppressor: NonMaxSuppressor,
        # box_fuser: BoxFuser,                          # TODO: implement BoxFuser for several feature detectors
        edema_classifier: EdemaClassifier,
        img_size: Tuple[int, int] = (1536, 1536),
        lung_extension: Tuple[int, int, int, int] = (50, 50, 50, 150),
    ) -> None:
        self.lung_segmenters = lung_segmenters
        self.feature_detectors = feature_detectors
        self.map_fuser = map_fuser
        self.mask_processor = mask_processor
        self.non_max_suppressor = non_max_suppressor
        # self.box_fuser = box_fuser                    # TODO: implement BoxFuser for several feature detectors
        self.edema_classifier = edema_classifier
        self.img_size = img_size
        self.lung_extension = lung_extension  # Tuple[left (x1), top (y1), right (x2), bottom (y2)]

    def predict(
        self,
        img_path: str,
        save_dir: str,
    ) -> pd.DataFrame:
        # Create a directory and copy an image into it
        img_name = Path(img_path).stem
        img_dir = os.path.join(save_dir, img_name)
        os.makedirs(img_dir, exist_ok=True)
        dst_path = os.path.join(img_dir, self.IMG_NAME)
        shutil.copy(img_path, dst_path)
        img_path = dst_path
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        # Segment lungs and output the probability segmentation maps
        for idx, lung_segmenter in enumerate(self.lung_segmenters):
            prob_map_ = lung_segmenter.predict(
                img=img,
                scale_output=True,
            )
            prob_map = cv2.resize(
                prob_map_,
                (img_width, img_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
            self.map_fuser.add_prob_map(prob_map)
            map_path = os.path.join(
                img_dir,
                f'{self.MAP_PREFIX}_{lung_segmenter.model_name}.png',
            )
            cv2.imwrite(map_path, prob_map)

        # Merge probability segmentation maps into a single map
        fused_map = self.map_fuser.conditional_probability_fusion(scale_output=True)
        fused_map_path = os.path.join(img_dir, self.MAP_NAME)
        cv2.imwrite(fused_map_path, fused_map)

        # Process the fused map and get the final segmentation mask
        mask_bin = self.mask_processor.binarize_image(image=fused_map)
        mask_smooth = self.mask_processor.smooth_mask(mask=mask_bin)
        mask_clean = self.mask_processor.remove_artifacts(mask=mask_smooth)
        mask_path = os.path.join(img_dir, self.MASK_NAME)
        cv2.imwrite(mask_path, mask_clean)

        # Extract the coordinates of the lungs and expand them if necessary
        lungs_metadata = compute_lungs_metadata(mask=mask_clean)
        lung_coords_ = (
            lungs_metadata['x1'],
            lungs_metadata['y1'],
            lungs_metadata['x2'],
            lungs_metadata['y2'],
        )
        lungs_coords = modify_lung_box(
            img_height=img_height,
            img_width=img_width,
            lung_coords=lung_coords_,
            lung_extension=self.lung_extension,
        )

        # Process image and mask which are used by an object detector
        img_crop = process_image(
            img=img,
            x1=lungs_coords[0],
            y1=lungs_coords[1],
            x2=lungs_coords[2],
            y2=lungs_coords[3],
            output_size=self.img_size,
        )
        img_crop_path = os.path.join(img_dir, self.IMG_CROP_NAME)
        cv2.imwrite(img_crop_path, img_crop)
        mask_crop = process_image(
            img=mask_clean,
            x1=lungs_coords[0],
            y1=lungs_coords[1],
            x2=lungs_coords[2],
            y2=lungs_coords[3],
            output_size=self.img_size,
        )
        mask_crop_path = os.path.join(img_dir, self.MASK_CROP_NAME)
        cv2.imwrite(mask_crop_path, mask_crop)

        # Recognize features and perform NMS
        df_dets = pd.DataFrame()
        for idx, feature_detector in enumerate(self.feature_detectors):
            dets = feature_detector.predict(img=img_crop)
            df_dets_ = feature_detector.process_detections(
                img_path=img_crop_path,
                detections=dets,
            )
            df_nms = self.non_max_suppressor.suppress_detections(df=df_dets_)
            # df_nms['Model ID'] = idx + 1        # FIXME (Anton): I think it is better to create a list of data frames
            df_dets = pd.concat([df_dets, df_nms])

        # TODO: Perform box fusion
        # df_out = self.box_fuser.fuse_detections(df=df_dets)
        # print(len(df_nms))

        # Assign an edema class to an image
        df_out = self.edema_classifier.classify(df=df_dets)

        return df_out


def modify_lung_box(
    img_height: int,
    img_width: int,
    lung_coords: Tuple[int, int, int, int],
    lung_extension: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    x1 = lung_coords[0] - lung_extension[0]
    y1 = lung_coords[1] - lung_extension[1]
    x2 = lung_coords[2] + lung_extension[2]
    y2 = lung_coords[3] + lung_extension[3]

    # Check if the box coordinates exceed image dimensions
    if x1 < 0:
        logging.warning(f'x1 = {x1} exceeds the left edge of the image = {0}')
    if y1 < 0:
        logging.warning(f'y1 = {y1} exceeds the top edge of the image = {0}')
    if x2 > img_width:
        logging.warning(f'x2 = {x2} exceeds the right edge of the image = {img_width}')
    if y2 > img_height:
        logging.warning(f'y2 = {y2} exceeds the bottom edge of the image = {img_height}')

    # Check if x2 is greater than x1 and y2 is greater than y1
    if x2 <= x1:
        logging.warning(
            f'x2 = {x2} is not greater than x1 = {x1}',
        )
    if y2 <= y1:
        logging.warning(
            f'y2 = {y2} is not greater than y1 = {y1}',
        )

    # Clip coordinates to image dimensions if necessary
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    return x1, y1, x2, y2


def compute_lungs_metadata(
    mask: np.ndarray,
) -> dict:
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lung_coords = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        x1, y1, x2, y2 = x, y, x + width, y + height
        lung_coords.append([x1, y1, x2, y2])

    x1_values, y1_values, x2_values, y2_values = zip(*lung_coords)
    x1, y1 = min(x1_values), min(y1_values)
    x2, y2 = max(x2_values), max(y2_values)

    feature_name = 'Lungs'
    lungs_metadata = {
        'Feature ID': FEATURE_MAP[feature_name],
        'Feature': feature_name,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
    }

    lungs_metadata.update(get_box_sizes(x1=x1, y1=y1, x2=x2, y2=y2))

    return lungs_metadata


def process_image(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    output_size: Tuple[int, int],
) -> np.ndarray:
    transform = A.Compose(
        [
            A.Crop(
                x_min=x1,
                y_min=y1,
                x_max=x2,
                y_max=y2,
                always_apply=True,
            ),
            A.LongestMaxSize(
                max_size=max(output_size),
                interpolation=4,
                always_apply=True,
            ),
            A.PadIfNeeded(
                min_width=output_size[0],
                min_height=output_size[1],
                position='center',
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                always_apply=True,
            ),
        ],
    )

    transformed = transform(image=img)
    img_out = transformed['image']

    return img_out
