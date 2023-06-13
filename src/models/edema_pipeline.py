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
        output_size: Tuple[int, int] = (1536, 1536),
        lung_extension: Tuple[int, int, int, int] = (50, 50, 50, 150),
        save_dir: str = 'data/interim_predict',
    ) -> None:
        self.seg_model_dirs = seg_model_dirs
        self.det_model_dirs = det_model_dirs
        self.output_size = output_size
        self.lung_extension = lung_extension  # Order: left (x1), top (y1), right (x2), bottom (y2)
        self.save_dir = save_dir

    def __call__(
        self,
        img_path: str,
    ) -> None:
        # Create a directory and copy an image into it
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
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
            image_height=img_height,
            image_width=img_width,
            lung_coords=lung_coords_,
            lung_extension=self.lung_extension,
        )

        # Process image that is used by an object detector
        self.process_image(
            img=img,
            x1=lungs_coords[0],
            y1=lungs_coords[1],
            x2=lungs_coords[2],
            y2=lungs_coords[3],
            output_size=self.output_size,
        )

        # TODO: Detection

        # TODO: Classification

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

    @staticmethod
    def _modify_lung_box(
        image_height: int,
        image_width: int,
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
        if x2 > image_width:
            logging.warning(f'x2 = {x2} exceeds the right edge of the image = {image_width}')
        if y2 > image_height:
            logging.warning(f'y2 = {y2} exceeds the bottom edge of the image = {image_height}')

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
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        return x1, y1, x2, y2


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
        output_size=(1536, 1536),
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
        edema_net(img_path)

    print('Complete')
