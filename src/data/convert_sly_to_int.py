import argparse
import logging
import os
from functools import partial
from pathlib import Path
from typing import List, Optional

import cv2
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from tqdm import tqdm

from settings import (
    EXCLUDE_DIRS,
    FIGURE_TYPE,
    INCLUDE_DIRS,
    INTERMEDIATE_SAVE_DIR,
    SUPERVISELY_DATASET_DIR,
)
from src.data.utils_sly import (
    CLASS_MAP,
    FIGURE_MAP,
    METADATA_COLUMNS,
    get_box_sizes,
    get_class_name,
    get_mask_points,
    get_object_box,
    get_tag_value,
    read_sly_project,
)

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_image(
    row: pd.Series,
    save_dir: str,
) -> dict:
    """Process a single image.

    Args:
        row: series with information about one image
        save_dir: directory where the output frontal images and metadata will be stored
    Returns:
        dictionary with information about one image
    """
    (img_path, ann_path), row = row
    logger.info(f'Cropping image {img_path}')

    img_stem = Path(img_path).stem
    subject_id, study_id, width_frontal, width_lateral = img_stem.split('_')
    img = cv2.imread(img_path)
    height = img.shape[0]
    width = int(width_frontal)
    img = img[0:height, 0:width]

    save_dir_img = os.path.join(save_dir, 'img')
    os.makedirs(save_dir_img, exist_ok=True)
    img_path = os.path.join(save_dir_img, f'{subject_id}_{study_id}.png')
    cv2.imwrite(img_path, img)

    return {
        'Image path': img_path,
        'Image name': Path(img_path).name,
        'Subject ID': subject_id,
        'Study ID': study_id,
        'Image width': width,
        'Image height': height,
    }


def process_annotation(
    row: pd.Series,
    img_info: dict,
) -> pd.DataFrame:
    """Process a single annotation.

    Args:
        row: series with information about one image
        img_info: dictionary with information about one image
    Returns:
        meta: dataframe with metadata for one image
    """
    logger.info('Preparing metadata')
    (img_path, ann_path), row = row
    meta = pd.DataFrame(columns=METADATA_COLUMNS)

    ann = sly.io.json.load_json_file(ann_path)
    class_name = get_class_name(ann)

    if len(ann['objects']) > 0 and class_name in [
        'Vascular congestion',
        'Interstitial edema',
        'Alveolar edema',
    ]:
        for obj in ann['objects']:
            logger.debug(f'Processing object {obj}')

            rp = get_tag_value(obj, tag_name='RP')
            mask_points = get_mask_points(obj)
            xy = get_object_box(obj)
            box = get_box_sizes(*xy.values())
            figure_name = obj['classTitle']

            obj_info = {
                'Figure ID': FIGURE_MAP[figure_name],
                'Figure': figure_name,
                'Source type': obj['geometryType'],
                'Reference type': FIGURE_TYPE[figure_name],
                'Match': int(obj['geometryType'] == FIGURE_TYPE[figure_name]),
                'RP': rp,
                'Class ID': CLASS_MAP[class_name],
                'Class': class_name,
            }
            obj_info.update(img_info)
            obj_info.update(xy)
            obj_info.update(box)
            obj_info.update(mask_points)
            meta = meta.append(obj_info, ignore_index=True)

    elif len(ann['objects']) == 0 and class_name == 'No edema':
        obj_info = {
            'Class ID': CLASS_MAP[class_name],
            'Class': class_name,
        }
        obj_info.update(img_info)
        meta = meta.append(obj_info, ignore_index=True)

    # else:
    #     logger.warning(f'No objects or classes available for image {Path(img_path).name}')

    return meta


def process_sample(
    row: pd.Series,
    save_dir: str,
) -> pd.DataFrame:
    """Process a single sample.

    Args:
        row: series with information about one image
        save_dir: directory where the output frontal images and metadata will be stored
    Returns:
        ann_info: dataframe with metadata for one image
    """
    img_info = process_image(row, save_dir)
    ann_info = process_annotation(row, img_info)

    return ann_info


def save_metadata(
    metadata: pd.DataFrame,
    save_dir: str,
) -> None:
    """Save metadata for an intermediate dataset.

    Args:
        metadata: dataframe with metadata for all images
        save_dir: directory where the output files will be saved
    Returns:
        None
    """
    metadata_path = os.path.join(save_dir, 'metadata.xlsx')
    logging.info(f'Saving metadata to {metadata_path}')
    metadata.sort_values(['Image path'], inplace=True)
    metadata.reset_index(drop=True, inplace=True)
    metadata.index += 1
    metadata.to_excel(
        metadata_path,
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )


def main(
    dataset_dir: str,
    save_dir: str,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> None:
    """Convert Supervisely dataset into Intermediate.

    Args:
        dataset_dir: a path to Supervisely dataset directory
        include_dirs: a list of subsets to include in the dataset
        exclude_dirs: a list of subsets to exclude from the dataset
        save_dir: directory where the output frontal images and metadata will be stored
    Returns:
        None
    """
    df = read_sly_project(
        dataset_dir=dataset_dir,
        include_dirs=include_dirs,
        exclude_dirs=exclude_dirs,
    )

    logger.info('Processing annotations')
    groups = df.groupby(['img_path', 'ann_path'])
    processing_func = partial(
        process_sample,
        save_dir=save_dir,
    )
    result = Parallel(n_jobs=-1)(
        delayed(processing_func)(group)
        for group in tqdm(groups, desc='Dataset conversion', unit=' image')
    )

    metadata = pd.concat(result)
    save_metadata(
        metadata=metadata,
        save_dir=save_dir,
    )
    logger.info('Complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervisely-to-Intermediate dataset conversion')
    parser.add_argument('--dataset_dir', default=SUPERVISELY_DATASET_DIR, type=str)
    parser.add_argument('--include_dirs', nargs='+', default=INCLUDE_DIRS, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=EXCLUDE_DIRS, type=str)
    parser.add_argument('--save_dir', default=INTERMEDIATE_SAVE_DIR, type=str)
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
        save_dir=args.save_dir,
    )
