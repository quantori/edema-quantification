import gc
import logging
import os
import shutil
from glob import glob
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch

from src.data.utils_sly import FEATURE_MAP, get_box_sizes
from src.models.feature_detector import FeatureDetector
from src.models.lung_segmenter import LungSegmenter
from src.models.map_fuser import MapFuser
from src.models.mask_processor import MaskProcessor
from src.models.non_max_suppressor import NonMaxSuppressor


class EdemaNet:
    """A network dedicated to processing X-ray images and predicting the stage of edema."""

    object_names = {
        'img': 'img.png',
        'mask': 'mask.png',
        'img_crop': 'img_crop.png',
        'mask_crop': 'mask_crop.png',
        'map': 'map.png',
        'map_prefix': 'map',
        'metadata': 'metadata.xlsx',
    }

    def __init__(
        self,
        lung_segmenters: List[LungSegmenter],
        feature_detectors: List[FeatureDetector],
        map_fuser: MapFuser,
        mask_processor: MaskProcessor,
        non_max_suppressor: NonMaxSuppressor,
        nms_method: str = 'soft',
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.7,
        output_size: Tuple[int, int] = (1536, 1536),
        lung_extension: Tuple[int, int, int, int] = (50, 50, 50, 150),
        save_dir: str = 'data/interim_predict',
    ) -> None:
        self.lung_segmenters = lung_segmenters
        self.feature_detectors = feature_detectors
        self.map_fuser = map_fuser
        self.mask_processor = mask_processor
        self.non_max_suppressor = non_max_suppressor
        assert nms_method in ['standard', 'soft'], f'Unknown fusion method: {nms_method}'
        self.nms_method = nms_method
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.output_size = output_size
        self.lung_extension = lung_extension  # Tuple[left (x1), top (y1), right (x2), bottom (y2)]
        self.save_dir = save_dir

    def predict(
        self,
        img_path: str,
    ) -> None:
        # Create a directory and copy an image into it
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        img_name = Path(img_path).stem
        img_dir = os.path.join(self.save_dir, img_name)
        os.makedirs(img_dir, exist_ok=True)
        dst_path = os.path.join(img_dir, self.object_names['img'])
        shutil.copy(img_path, dst_path)

        # Segment lungs and output the probability segmentation maps
        for lung_segmenter in self.lung_segmenters:
            prob_map_ = lung_segmenter.predict(
                img=img,
                scale_output=True,
            )
            cv2.resize(
                prob_map_,
                (img_width, img_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
            # map_path = os.path.join(
            #     self.save_dir,
            #     f'{self.object_names["map_prefix"]}_{model_name}.png',
            # )
            # cv2.imwrite(map_path, prob_map)

        # Merge probability segmentation maps into a single map
        # TODO: add maps to map_fuser
        # for map_path in map_paths:
        #     self.map_fuser.add_prob_map(map_path)
        fused_map = self.map_fuser.conditional_probability_fusion(scale_output=True)

        # TODO: Remove
        self.fuse_maps(
            img_dir=img_dir,
        )

        # Process the fused map and get the final segmentation mask
        self.process_fused_map(
            img_dir=img_dir,
        )
        mask_bin = self.mask_processor.binarize_image(image=fused_map)
        mask_smooth = self.mask_processor.smooth_mask(mask=mask_bin)
        self.mask_processor.remove_artifacts(mask=mask_smooth)

        # Extract the coordinates of the lungs and expand them if necessary
        lungs_metadata = self.compute_lungs_metadata(
            img_dir=img_dir,
        )
        lung_coords_ = (
            lungs_metadata['x1'],
            lungs_metadata['y1'],
            lungs_metadata['x2'],
            lungs_metadata['y2'],
        )
        lungs_coords = self._modify_lung_box(
            img_height=img_height,
            img_width=img_width,
            lung_coords=lung_coords_,
            lung_extension=self.lung_extension,
        )

        # Process image that is used by an object detector
        self.process_image(
            img_dir=img_dir,
            x1=lungs_coords[0],
            y1=lungs_coords[1],
            x2=lungs_coords[2],
            y2=lungs_coords[3],
            output_size=self.output_size,
        )

        # Recognize features and return them as a dataframe
        # TODO: move model loading to __init__
        df_dets = self.detect_features(
            img_dir=img_dir,
        )

        # Perform Soft Non-Maximum Suppression
        # TODO: Filter dataframes by class conference independently
        box_fuser = NonMaxSuppressor(
            method=self.nms_method,
            sigma=0.1,
            iou_threshold=self.iou_threshold,
            conf_threshold=self.conf_threshold,
        )
        df_nms = box_fuser.fuse_detections(df=df_dets)

        # TODO: Process detected features and assign stage of edema
        self.classify_edema_stage(
            df=df_nms,
            img_dir=img_dir,
        )

    def segment_lungs(
        self,
        img_path: str,
        save_dir: str,
    ) -> None:
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        for model_dir in self.seg_model_dirs:
            # Initialize segmentation model
            model_name = Path(model_dir).name
            model = LungSegmenter(
                model_dir=model_dir,
                device='auto',
            )

            # Retrieve and save a probability segmentation map
            prob_map_ = model.predict(
                img=img,
                scale_output=True,
            )
            prob_map = cv2.resize(
                prob_map_,
                (img_width, img_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
            map_path = os.path.join(
                save_dir,
                f'{self.object_names["map_prefix"]}_{model_name}.png',
            )
            cv2.imwrite(map_path, prob_map)

            # Run the garbage collector and release all unused cached memory
            del model
            gc.collect()
            torch.cuda.empty_cache()

    def fuse_maps(
        self,
        img_dir,
    ):
        # Retrieve paths to probability maps
        search_pattern = os.path.join(img_dir, f'{self.object_names["map_prefix"]}*.png')
        map_paths = glob(search_pattern)

        # Read probability maps and then merge them into one
        fuser = MapFuser()
        for map_path in map_paths:
            fuser.add_prob_map(map_path)
        fused_map = fuser.conditional_probability_fusion()
        fused_map = (fused_map * 255.0).astype(np.uint8)

        # Save fused probability map
        fused_map_path = os.path.join(img_dir, self.object_names['map'])
        cv2.imwrite(fused_map_path, fused_map)

    def process_fused_map(
        self,
        img_dir: str,
    ):
        # Retrieve path to the fused map
        fused_map_path = os.path.join(img_dir, self.object_names['map'])
        fused_map = cv2.imread(fused_map_path, cv2.IMREAD_GRAYSCALE)

        processor = MaskProcessor()
        mask_bin = processor.binarize_image(image=fused_map)
        mask_smooth = processor.smooth_mask(mask=mask_bin)
        mask_clean = processor.remove_artifacts(mask=mask_smooth)

        # Store lung segmentation mask
        mask_path = os.path.join(img_dir, self.object_names['mask'])
        cv2.imwrite(mask_path, mask_clean)

    def compute_lungs_metadata(
        self,
        img_dir: str,
    ) -> dict:
        mask_path = os.path.join(img_dir, 'mask.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
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
        self,
        img_dir: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        output_size: Tuple[int, int],
    ):
        img_path = os.path.join(img_dir, self.object_names['img'])
        mask_path = os.path.join(img_dir, self.object_names['mask'])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

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

        transformed = transform(image=img, mask=mask)
        img_out = transformed['image']
        mask_out = transformed['mask']
        save_img_path = os.path.join(img_dir, self.object_names['img_crop'])
        save_mask_path = os.path.join(img_dir, self.object_names['mask_crop'])
        cv2.imwrite(save_img_path, img_out)
        cv2.imwrite(save_mask_path, mask_out)

    @staticmethod
    def _modify_lung_box(
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

    def detect_features(
        self,
        img_dir: str,
    ) -> pd.DataFrame:
        # FIXME: currently only one models is supported
        img_path = os.path.join(img_dir, self.object_names['img_crop'])
        for model_dir in self.det_model_dirs:
            model = FeatureDetector(
                model_dir=model_dir,
                batch_size=1,
                conf_threshold=0.01,
                device='auto',
            )
            dets = model([img_path])
            df_dets = model.process_detections(
                img_paths=[img_path],
                detections=dets,
            )

        return df_dets

    def classify_edema_stage(
        self,
        df: pd.DataFrame,
        img_dir: str,
    ) -> None:
        save_path = os.path.join(img_dir, self.object_names['metadata'])
        df.index += 1
        df.to_excel(
            save_path,
            sheet_name='Metadata',
            index=True,
            index_label='ID',
        )
