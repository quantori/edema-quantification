import os
import logging
import argparse

import cv2
from typing import List, Optional
from pathlib import Path

import pandas as pd
import supervisely_lib as sly

from tools.utils_sly import (
    CLASS_MAP,
    FIGURE_MAP,
    read_sly_project,
    get_class_name,
    get_tag_value,
    get_object_box,
    get_box_sizes,
    METADATA_COLUMNS,
    get_object_points_mask,
    ANNOTATION_COLUMNS,
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


def process_image(row: pd.Series, save_dir_ann, save_dir_img_frontal, save_dir_img_lateral):
    logger.info(f'Cropping image {row.img_path}')

    original_img_name = os.path.basename(row.img_path)
    subject_id, study_id, width_frontal, width_lateral, ext = original_img_name.replace(
        '.', '_'
    ).split('_')
    img = cv2.imread(row.img_path)
    height = img.shape[0]
    width = img.shape[1]
    img_frontal = img[0:height, 0 : int(width_frontal)]
    img_lateral = img[0:height, int(width_frontal) : width]
    img_frontal_path = os.path.join(save_dir_img_frontal, f'{subject_id}_{study_id}.{ext}')
    cv2.imwrite(img_frontal_path, img_frontal)
    cv2.imwrite(os.path.join(save_dir_img_lateral, f'{subject_id}_{study_id}.{ext}'), img_lateral)

    logger.info('Preparing metadata and annotation')
    img_metadata = pd.DataFrame(columns=METADATA_COLUMNS)
    annotation = pd.DataFrame(columns=ANNOTATION_COLUMNS)

    ann = sly.io.json.load_json_file(row.ann_path)
    class_name = get_class_name(ann)

    if len(ann['objects']) == 0:
        logger.warning(f'There is no objects!')
    else:
        for obj in ann['objects']:
            logger.info(f'Processing object {obj}')

            rp = get_tag_value(obj, tag_name='RP')
            mask_points = get_object_points_mask(obj)
            xy = get_object_box(obj)
            box = get_box_sizes(*xy.values())
            figure_name = obj['classTitle']

            annotation_info = {
                'edema id': CLASS_MAP[class_name],
                'figure id': FIGURE_MAP[figure_name],
            }
            annotation_info.update(xy)
            annotation = annotation.append(annotation_info, ignore_index=True)
            new_annotation_name = f'{subject_id}_{study_id}.csv'
            logging.info(f'Saving annotation {new_annotation_name}')
            annotation.to_csv(
                os.path.join(save_dir_ann, new_annotation_name),
                header=False,
                index=False,
                sep=' ',
            )

            image_info = {
                'Image path': img_frontal_path,
                'Subject ID': subject_id,
                'Study id': study_id,
                'Image width': width,
                'Image height': height,
                'Figure': figure_name,
                'RP': rp,
                'Class ID': CLASS_MAP[class_name],
                'Class': class_name,
            }
            image_info.update(xy)
            image_info.update(box)
            image_info.update(mask_points)
            img_metadata = img_metadata.append(image_info, ignore_index=True)

    return img_metadata


def main(
    save_dir: str,
    dataset_dir: str,
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

    logger.info(f'Reading a sly project {dataset_dir}')
    sly_project = read_sly_project(
        dataset_dir=dataset_dir,
        include_dirs=include_dirs,
        exclude_dirs=exclude_dirs,
    )

    logger.info(f'Creating img and ann directories in {save_dir}')
    save_dir_img_frontal = os.path.join(save_dir, 'img', 'frontal')
    os.makedirs(save_dir_img_frontal, exist_ok=True)
    save_dir_img_lateral = os.path.join(save_dir, 'img', 'lateral')
    os.makedirs(save_dir_img_lateral, exist_ok=True)
    save_dir_ann = os.path.join(save_dir, 'ann')
    os.makedirs(save_dir_ann, exist_ok=True)

    logger.info('Preparing metadata and annotations')
    all_metadata = pd.DataFrame(columns=METADATA_COLUMNS)
    for idx, row in sly_project.iterrows():
        all_metadata = all_metadata.append(
            process_image(row, save_dir_ann, save_dir_img_frontal, save_dir_img_lateral),
            ignore_index=True,
        )

    # TODO (@irina.ryndova): parallel processing (optional)
    metadata_path = os.path.join(save_dir, 'metadata.xlsx')
    logging.info(f'Saving {metadata_path}')
    all_metadata.index += 1
    all_metadata.to_excel(
        metadata_path,
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Supervisely dataset')
    parser.add_argument('--dataset_dir', default='dataset/MIMIC-CXR-Edema-SLY', type=str)
    parser.add_argument('--include_dirs', nargs='+', default=[], type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=[], type=str)  # TODO nargs?
    parser.add_argument('--save_dir', default='dataset/MIMIC-CXR-Edema-Convert', type=str)
    args = parser.parse_args()

    main(
        save_dir=args.save_dir,
        dataset_dir=args.dataset_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
    )
