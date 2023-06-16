import logging
import os
from pathlib import Path
from typing import List

import cv2
import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import get_file_list
from src.data.utils_sly import (
    FEATURE_MAP,
    FEATURE_TYPE,
    METADATA_COLUMNS,
    convert_mask_to_base64,
    get_box_sizes,
)
from src.models.map_fuser import MapFuser
from src.models.mask_processor import MaskProcessor

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def extract_lungs_metadata(
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

    mask_crop = mask[y1:y2, x1:x2]
    mask_encoded = convert_mask_to_base64(mask_crop)

    feature_name = 'Lungs'
    lungs_info = {
        'Feature ID': FEATURE_MAP[feature_name],
        'Feature': feature_name,
        'Source type': 'bitmap',
        'Reference type': FEATURE_TYPE[feature_name],
        'Match': 1,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'Mask': mask_encoded,
    }

    lungs_info.update(get_box_sizes(x1=x1, y1=y1, x2=x2, y2=y2))

    return lungs_info


def process_prob_maps(
    img_paths: List[str],
    save_dir: str,
) -> dict:
    # Fuse segmentation probability maps
    fuser = MapFuser()
    for img_path in img_paths:
        prob_map = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        fuser.add_prob_map(prob_map)
    fused_map = fuser.conditional_probability_fusion(scale_output=True)

    # Process obtained fused map
    processor = MaskProcessor()
    mask_bin = processor.binarize_image(image=fused_map)
    mask_smooth = processor.smooth_mask(mask=mask_bin)
    mask_clean = processor.remove_artifacts(mask=mask_smooth)

    # Save the fused map and its mask
    img_name = Path(img_paths[0]).name
    map_dir = os.path.join(save_dir, 'map')
    mask_dir = os.path.join(save_dir, 'mask')
    os.makedirs(map_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    map_path = os.path.join(map_dir, f'{img_name}')
    mask_path = os.path.join(mask_dir, f'{img_name}')
    cv2.imwrite(map_path, fused_map)
    cv2.imwrite(mask_path, mask_clean)

    # Extract lungs metadata
    img_stem = Path(img_path).stem
    subject_id, study_id = img_stem.split('_')
    map_height, map_width = fused_map.shape[:2]
    map_ratio = map_height / map_width
    lungs_info = {
        'Image name': img_name,
        'Subject ID': subject_id,
        'Study ID': study_id,
        'Image width': map_width,
        'Image height': map_height,
        'Image ratio': map_ratio,
        'View': 'Frontal',
    }
    lung_coords = extract_lungs_metadata(mask=mask_clean)
    lungs_info.update(lung_coords)

    return lungs_info


def reorder_image_paths(
    input_lists: List[List[str]],
) -> List[List[str]]:
    # Check if all input lists have the same number of elements
    num_elements = len(input_lists[0])
    assert all(
        len(lst) == num_elements for lst in input_lists
    ), 'All input lists must have the same number of elements'

    # Use zip() to iterate over the elements at the same index from the input lists
    output_list = []
    for items in zip(*input_lists):
        # Check if all file paths have the same basename
        basenames = [os.path.basename(path) for path in items]
        assert all(
            basename == basenames[0] for basename in basenames
        ), 'All file paths must have the same basename'
        output_list.append(list(items))

    return output_list


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='fuse_maps',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    log.info(f'Models used...............: {cfg.model_names}')
    log.info('')

    # Get list of paths to images
    img_path_sets: List[List[str]] = []
    for model_name in cfg.model_names:
        img_dir = os.path.join(cfg.img_dir, f'{model_name}')
        img_paths_ = get_file_list(
            src_dirs=img_dir,
            ext_list=[
                '.png',
                '.jpg',
                '.jpeg',
                '.bmp',
            ],
        )
        img_path_sets.append(img_paths_)
        log.info(f'{len(img_paths_)} images are used for {model_name}')

    # Reorder image paths for multiprocessing
    img_path_sets = reorder_image_paths(img_path_sets)

    # Process segmentation probability maps
    lung_info = Parallel(n_jobs=1)(
        delayed(process_prob_maps)(img_path_set, cfg.save_dir)
        for img_path_set in tqdm(img_path_sets, desc='Processing')
    )

    # Create metadata
    metadata = pd.DataFrame(columns=METADATA_COLUMNS)
    metadata = metadata.append(lung_info, ignore_index=True)

    # Save metadata
    metadata_path = os.path.join(cfg.save_dir, 'metadata.xlsx')
    log.info(f'Saving metadata to {metadata_path}')
    metadata.sort_values(['Image name'], inplace=True)
    metadata.reset_index(drop=True, inplace=True)
    metadata.index += 1
    metadata.to_excel(
        metadata_path,
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
