import gc
import logging
import os
import shutil
from glob import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.data.utils import get_file_list
from src.data.utils_sly import FEATURE_MAP, get_box_sizes
from src.models.lung_segmenter import LungSegmenter
from src.models.map_fuser import MapFuser
from src.models.mask_processor import MaskProcessor


class EdemaNet:
    """A pipeline dedicated to processing X-ray images and predicting the stage of edema."""

    def __init__(
        self,
        seg_model_dirs: List[str],
        det_model_dirs: List[str],
        lung_extension: Tuple[int, int, int, int] = (50, 50, 50, 150),
        save_dir: str = 'data/interim_predict',
    ) -> None:
        self.seg_model_dirs = seg_model_dirs
        self.det_model_dirs = det_model_dirs
        self.lung_extension = lung_extension
        self.save_dir = save_dir

    def __call__(
        self,
        img_path: str,
    ):
        # Create a directory and copy an image into it
        img_name = Path(img_path).stem
        img_dir = os.path.join(self.save_dir, img_name)
        os.makedirs(img_dir, exist_ok=True)
        shutil.copy(img_path, img_dir)

        # Lung segmentation
        self.segment_lungs(
            img_path=img_path,
            save_dir=img_dir,
        )

        # Mask fusion
        self.fuse_maps(
            img_dir=img_dir,
        )

        # Process fused map
        self.process_fused_map(
            img_dir=img_dir,
        )

        # Extract lungs metadata
        self.compute_lungs_metadata(
            img_dir=img_dir,
        )

        # TODO: Crop + add box extension

        # TODO: Detection

        # TODO: Classification

        return 1

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
            prob_map_ = model(
                img=img,
                scale_output=True,
            )
            prob_map = cv2.resize(
                prob_map_,
                (img_width, img_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
            map_path = os.path.join(save_dir, f'prob_map_{model_name}.png')
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
        prefix = 'prob_map'
        search_pattern = os.path.join(img_dir, f'{prefix}*.png')
        map_paths = glob(search_pattern)

        # Read probability maps and then merge them into one
        fuser = MapFuser()
        for map_path in map_paths:
            fuser.add_prob_map(map_path)
        fused_map = fuser.conditional_probability_fusion()
        fused_map = (fused_map * 255.0).astype(np.uint8)

        # Save fused probability map
        fused_map_path = os.path.join(img_dir, 'prob_map_fused.png')
        cv2.imwrite(fused_map_path, fused_map)

    def process_fused_map(
        self,
        img_dir: str,
    ):
        # Retrieve path to the fused map
        fused_map_path = os.path.join(img_dir, 'prob_map_fused.png')
        fused_map = cv2.imread(fused_map_path, cv2.IMREAD_GRAYSCALE)

        processor = MaskProcessor()
        mask_bin = processor.binarize_image(image=fused_map)
        mask_smooth = processor.smooth_mask(mask=mask_bin)
        mask_clean = processor.remove_artifacts(mask=mask_smooth)

        # Store lung segmentation mask
        mask_path = os.path.join(img_dir, 'mask.png')
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
        lungs_info = {
            'Feature ID': FEATURE_MAP[feature_name],
            'Feature': feature_name,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
        }

        lungs_info.update(get_box_sizes(x1=x1, y1=y1, x2=x2, y2=y2))

        return lungs_info


if __name__ == '__main__':
    data_dir = 'data/interim'
    results_dir = f'{data_dir}_predict'
    edema_net = EdemaNet(
        seg_model_dirs=[
            'models/lung_segmentation/DeepLabV3',
            'models/lung_segmentation/FPN',
            'models/lung_segmentation/MAnet',
        ],
        det_model_dirs=[
            'models/feature_detection/FasterRCNN',
        ],
        lung_extension=(50, 50, 50, 150),
        save_dir=results_dir,
    )

    img_paths = get_file_list(
        src_dirs=data_dir,
        ext_list=[
            '.png',
            '.jpg',
            '.jpeg',
            '.bmp',
        ],
    )
    logging.info(f'Number of images..........: {len(img_paths)}')

    for img_path in img_paths:
        print(f'Image: {Path(img_path).stem}')
        result = edema_net(img_path)

    print('Complete')
