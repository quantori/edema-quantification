import gc
import logging
import os
from pathlib import Path
from typing import List

import cv2
import torch

from src.data.utils import get_file_list
from src.models.lung_segmenter import LungSegmenter


class EdemaPipeline:
    """A pipeline dedicated to processing X-ray images and predicting the stage of edema."""

    def __init__(
        self,
        seg_model_dirs: List[str],
        det_model_dirs: List[str],
        save_dir: str,
    ) -> None:
        self.save_dir = save_dir
        self.seg_model_dirs = seg_model_dirs
        self.det_model_dirs = det_model_dirs

    def __call__(
        self,
        img_path: str,
    ):
        # Create a directory for a given image
        img_name = Path(img_path).name
        self.img_dir = os.path.join(self.save_dir, img_name)
        os.makedirs(self.img_dir)

        # TODO: Segmentation

        # TODO: Mask fusion

        # TODO: Crop

        # TODO: Detection

        # TODO: Classification

        return 1

    def segment_lungs(
        self,
        img_path: str,
    ):
        img_name = Path(img_path).name
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        for model_dir in self.seg_model_dirs:
            # Initialize segmentation model
            Path(model_dir).name
            model = LungSegmenter(
                model_dir=model_dir,
                device='auto',
            )

            # Retrieve and save a probability segmentation map
            map_ = model(
                img=img,
                scale_output=True,
            )
            map = cv2.resize(map_, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
            map_path = os.path.join(img_dir, img_name)
            cv2.imwrite(map_path, map)

            # Run the garbage collector and release all unused cached memory
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    img_dir = 'data/coco/test'

    edema_pipeline = EdemaPipeline(
        seg_model_dirs=['models/lung_segmentation/MAnet'],
        det_model_dirs=['models/feature_detection/FasterRCNN'],
        save_dir='eval_results',
    )

    img_paths = get_file_list(
        src_dirs=img_dir,
        ext_list=[
            '.png',
            '.jpg',
            '.jpeg',
            '.bmp',
        ],
    )
    logging.info(f'Number of images..........: {len(img_paths)}')

    for img_path in img_paths:
        result = edema_pipeline(img_path)

    print('Complete')
