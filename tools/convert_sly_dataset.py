import os
import logging
import argparse
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed

import cv2
import pandas as pd
from tqdm import tqdm
import supervisely_lib as sly
from typing import List, Optional, Tuple


from tools.utils_sly import (
    CLASS_MAP,
    FIGURE_MAP,
    METADATA_COLUMNS,
    ANNOTATION_COLUMNS,
    read_sly_project,
    get_class_name,
    get_tag_value,
    get_object_box,
    get_box_sizes,
    get_mask_points,
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
    save_dir_img: str,
) -> dict:
    """

    Args:
        row: series with information about one image
        save_dir_img: directory where the output frontal images will be saved

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
    img_frontal = img[0:height, 0:width]
    img_frontal_path = os.path.join(save_dir_img, f'{subject_id}_{study_id}.png')
    cv2.imwrite(img_frontal_path, img_frontal)

    return {
        'Image path': img_frontal_path,
        'Subject ID': subject_id,
        'Study ID': study_id,
        'Image width': width,
        'Image height': height,
    }


def process_annotation(
    row: pd.Series,
    save_dir_ann: str,
    img_info: dict,
) -> pd.DataFrame:
    """

    Args:
        row: series with information about one image
        save_dir_ann: directory where the output annotations will be saved
        img_info: dictionary with information about one image

    Returns:
        dataframe with metadata for one image
    """
    logger.info('Preparing metadata and annotation')
    (img_path, ann_path), row = row
    img_metadata = pd.DataFrame(columns=METADATA_COLUMNS)
    ann_metadata = pd.DataFrame(columns=ANNOTATION_COLUMNS)

    ann = sly.io.json.load_json_file(ann_path)
    class_name = get_class_name(ann)

    if len(ann['objects']) == 0:
        logger.warning(f'No objects for image {Path(img_path).name}')
    else:
        for obj in ann['objects']:
            logger.debug(f'Processing object {obj}')

            rp = get_tag_value(obj, tag_name='RP')
            mask_points = get_mask_points(obj)
            xy = get_object_box(obj)
            box = get_box_sizes(*xy.values())
            figure_name = obj['classTitle']

            ann_info = {
                'Class ID': CLASS_MAP[class_name],
                'Figure ID': FIGURE_MAP[figure_name],
                'RP': rp,
            }
            if xy['x1'] >= img_info['Image width']:
                continue
            ann_info.update(xy)
            ann_metadata = ann_metadata.append(ann_info, ignore_index=True)
            ann_name = f'{img_info["Subject ID"]}_{img_info["Study ID"]}.txt'
            ann_path = os.path.join(save_dir_ann, ann_name)
            logging.debug(f'Saving annotation {ann_name}')
            ann_metadata.to_csv(
                ann_path,
                header=False,
                index=False,
                sep='\t',
            )

            obj_info = {
                'Figure': figure_name,
                'RP': rp,
                'Class ID': CLASS_MAP[class_name],
                'Class': class_name,
            }
            obj_info.update(img_info)
            obj_info.update(xy)
            obj_info.update(box)
            obj_info.update(mask_points)
            img_metadata = img_metadata.append(obj_info, ignore_index=True)
    return img_metadata


def process_sample(
    row: pd.Series,
    save_dir_ann: str,
    save_dir_img: str,
) -> pd.DataFrame:
    """

    Args:
        row: series with information about one image
        save_dir_ann: directory where the output annotation will be saved
        save_dir_img: directory where the output frontal images will be saved

    Returns:
        dataframe with metadata for one image
    """
    img_info = process_image(row, save_dir_img)
    ann_info = process_annotation(row, save_dir_ann, img_info)

    return ann_info


def create_save_dirs(
    save_dir: str,
) -> Tuple[str, str]:
    """

    Args:
        save_dir: directory where the output files will be saved

    Returns:
        tuple with directories where the output frontal images and annotations will be saved
    """
    logger.info(f'Creating img and ann directories in {save_dir}')

    save_dir_img = os.path.join(save_dir, 'img')
    os.makedirs(save_dir_img, exist_ok=True)

    save_dir_ann = os.path.join(save_dir, 'ann')
    os.makedirs(save_dir_ann, exist_ok=True)

    return save_dir_img, save_dir_ann


def save_metadata(
    metadata: pd.DataFrame,
    save_dir: str,
) -> None:
    """

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
    """

    Args:
        dataset_dir: a path to Supervisely dataset directory
        include_dirs: a list of subsets to include in the dataset
        exclude_dirs: a list of subsets to exclude from the dataset
        save_dir: directory where the output files will be saved

    Returns:
        None
    """

    df = read_sly_project(
        dataset_dir=dataset_dir,
        include_dirs=include_dirs,
        exclude_dirs=exclude_dirs,
    )

    save_dir_img, save_dir_ann = create_save_dirs(save_dir=save_dir)

    logger.info('Processing annotations')
    groups = df.groupby(['img_path', 'ann_path'])
    processing_func = partial(
        process_sample,
        save_dir_ann=save_dir_ann,
        save_dir_img=save_dir_img,
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
    parser = argparse.ArgumentParser(description='Convert Supervisely dataset')
    parser.add_argument('--dataset_dir', default='dataset/MIMIC-CXR-Edema-Supervisely', type=str)
    parser.add_argument('--include_dirs', nargs='+', default=[], type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=[], type=str)
    parser.add_argument('--save_dir', default='dataset/MIMIC-CXR-Edema-Intermediate', type=str)
    args = parser.parse_args()

    main(
        save_dir=args.save_dir,
        dataset_dir=args.dataset_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
    )
