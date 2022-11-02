import os
import logging
import argparse
from PIL import Image
from typing import List
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


# TODO: I would suggest processing images along with annotations
def crop_images(
        dataset_img_dir: str,
        save_dir: str,
) -> None:
    """

    Args:
        dataset_img_dir: a path to the MIMIC-CXR dataset images
        save_dir: directory where the output files will be saved

    Returns:
        None
    """
    logger.info('Cropping images')

    save_img_dir = os.path.join(save_dir, 'img')
    os.makedirs(save_img_dir, exist_ok=True)

    img_list = os.listdir(dataset_img_dir)

    top = 0
    left = 0
    for img_name in img_list:
        img = Image.open(os.path.join(dataset_img_dir, img_name))
        subject_id, study_id, left_width, right_width, ext = img_name.replace(
            '.', '_'
        ).split('_')
        right = int(left_width)
        bottom = img.height

        img = img.crop((left, top, right, bottom))
        img.save(os.path.join(save_img_dir, f'{subject_id}_{study_id}.{ext}'))


def prepare_metadata_annotations(
        dataset_ann_dir: str,
        save_dir: str,
) -> None:
    """

    Args:
        dataset_ann_dir: a path to the MIMIC-CXR dataset annotations
        save_dir: directory where the output files will be saved

    Returns:
        None
    """
    logger.info('Preparing metadata and annotations')

    save_ann_dir = os.path.join(save_dir, 'ann')
    os.makedirs(save_ann_dir, exist_ok=True)

    metadata = pd.DataFrame(
        columns=[
            'Image path',
            'Subject ID',
            'Study id',
            'Image width',
            'Image height',
            'Figure',
            'x1',
            'y1',
            'x2',
            'y2',
            'xc',
            'yc',
            'Box width',
            'Box height',
            'RP',
            # 'Mask',           # TODO: add mask string
            # 'Points',         # TODO: add mask string
            # 'Class ID',       # TODO: add class id
            'Class',
        ]
    )

    ann_list = os.listdir(dataset_ann_dir)
    for ann_name in ann_list:
        logger.info(f'Processing annotation {ann_name}')
        annotation = pd.DataFrame(
            columns=['edema id', 'figure id', 'x1', 'y1', 'x2', 'y2']
        )

        ann = sly.io.json.load_json_file(os.path.join(dataset_ann_dir, ann_name))

        class_name = get_class_name(ann)

        subject_id, study_id, width, right_width, ext, _ = ann_name.replace(
            '.', '_'
        ).split('_')
        height = ann['size']['height']
        cropped_img_path = os.path.join(
            save_dir, 'img', f'{subject_id}_{study_id}.{ext}'
        )

        if len(ann['objects']) == 0:
            logger.warning(f'There is no objects!')
            continue

        for obj in ann['objects']:
            logger.info(f'Processing object {obj}')

            # rp = get_tag_value(obj, tag_name='RP')             # TODO: implement extraction tag_value by tag_name (in our case 'RP'), uncomment when fixed
            xy = get_object_box(obj)
            box = get_box_sizes(*xy.values())
            figure_name = obj['classTitle']

            annotation_info = {
                'edema id': CLASS_MAP[class_name],
                'figure id': FIGURE_MAP[figure_name],
            }
            annotation_info.update(xy)
            annotation = annotation.append(annotation_info, ignore_index=True)

            image_info = {
                'Image path': cropped_img_path,
                'Subject ID': subject_id,
                'Study id': study_id,
                'Image width': width,
                'Image height': height,
                'Figure': figure_name,
                'RP': rp,
                'Class': class_name,
            }
            image_info.update(xy)
            image_info.update(box)
            metadata = metadata.append(image_info, ignore_index=True)

        new_annotation_name = f'{subject_id}_{study_id}.csv'
        logging.info(f'Saving annotation {new_annotation_name}')
        annotation.to_csv(
            os.path.join(save_ann_dir, new_annotation_name),
            header=False,
            index=False,
            sep=' ',
        )

    # TODO: create a dataframe and save metadata as an XLSX file
    # df.index += 1
    # df.to_excel(
    #     save_path,
    #     sheet_name='Metadata',
    #     index=True,
    #     index_label='ID',
    # )
    logging.info('Saving metadata.csv')
    metadata.to_csv(os.path.join(save_dir, f'metadata.csv'))


def main(
        dataset_dir: str,
        include_dirs: List[str],
        exclude_dirs: List[str],
        save_dir: str,
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
    # TODO: remove
    dataset_ann_dir = os.path.join(dataset_dir, 'DS1', 'ann')
    dataset_img_dir = os.path.join(dataset_dir, 'DS1', 'img')

    # TODO (@irina.ryndova): remove it as SLY automatically checks the structure of the dataset implemented in read_sly_project
    # check_dataset_dirs(dataset_dir, dataset_ann_dir, dataset_img_dir)

    df = read_sly_project(
        dataset_dir=dataset_dir,
        include_dirs=include_dirs,
        exclude_dirs=exclude_dirs,
    )

    # TODO: process images and annotations together
    # TODO: Allocate dataframes/dicts per each image
    # for idx, row in df.iterrows():
    #     print(row)
    #     # process_image(...)

    # TODO: Create final dataframe
    # TODO (@irina.ryndova): parallel processing (optional)
    # crop_images(dataset_img_dir, save_dir)

    prepare_metadata_annotations(dataset_ann_dir, save_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert Supervisely dataset')
    parser.add_argument('--dataset_dir', default='dataset/MIMIC-CXR-Edema-SLY', type=str)
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--save_dir', default='dataset/MIMIC-CXR-Edema-Convert', type=str)
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
        save_dir=args.save_dir,
    )
