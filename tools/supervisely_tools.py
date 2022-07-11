import os
import zlib
import json
import base64
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervisely_lib as sly

logging.basicConfig(level=logging.INFO)


def read_supervisely_project(sly_project_dir: str,
                             included_datasets: Optional[List[str]] = None,
                             excluded_datasets: Optional[List[str]] = None) -> Tuple[List[str], List[str], List[str]]:
    img_paths: List = []
    ann_paths: List = []
    dataset_names: List = []

    logging.debug('Processing of {:s}...'.format(sly_project_dir))
    assert os.path.exists(sly_project_dir) and os.path.isdir(sly_project_dir), 'Wrong path: {:s}'.format(
        sly_project_dir)
    project_fs = sly.Project(sly_project_dir, sly.OpenMode.READ)

    for dataset_fs in project_fs:
        dataset_name = dataset_fs.name

        if included_datasets and dataset_name not in included_datasets:
            logging.debug('Skip {:s} because it is not in the include_datasets list'.format(dataset_name))
            continue

        if excluded_datasets and dataset_name in excluded_datasets:
            logging.debug('Skip {:s} because it is in the exclude_datasets list'.format(dataset_name))
            continue

        dataset_names.append(dataset_name)
        for item_name in dataset_fs:
            img_path, ann_path = dataset_fs.get_item_paths(item_name)
            img_paths.append(img_path)
            ann_paths.append(ann_path)

    return img_paths, ann_paths, dataset_names


def convert_base64_to_image(s: str) -> np.ndarray:
    """
    The function convert_base64_to_image converts a base64 encoded string to a numpy array
    :param s: string
    :return: numpy array
    """
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)

    img_decoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if (len(img_decoded.shape) == 3) and (img_decoded.shape[2] >= 4):
        mask = img_decoded[:, :, 3].astype(np.uint8)  # 4-channel images
    elif len(img_decoded.shape) == 2:
        mask = img_decoded.astype(np.uint8)  # flat 2D mask
    else:
        raise RuntimeError('Wrong internal mask format.')
    return mask


def convert_ann_to_mask(ann_path: str,
                        filter_mask: bool = True) -> np.ndarray:
    with open(ann_path) as json_file:
        data = json.load(json_file)

    img_height = data['size']['height']
    img_width = data['size']['width']

    try:
        is_normal = True if data['tags'][0]['name'] == 'Normal' else False
    except Exception:
        is_normal = False

    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Objects with masks for example COVID-19 or Lungs
    if not is_normal:
        for obj in data['objects']:
            if obj['classTitle'] == class_name:
                encoded_bitmap = obj['bitmap']['data']
                _mask = convert_base64_to_image(s=encoded_bitmap)

                # Be careful with morphology filtering. Use the following approximate kernel sizes:
                # kernel_size = int(0.05 * min_dim) for big masks   (size > 1000x1000)
                # kernel_size = int(0.01 * min_dim) for small masks (size > 500x500)
                if filter_mask:
                    min_dim = min(_mask.shape)
                    kernel_size = max(int(0.01 * min_dim), 1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    _mask = cv2.morphologyEx(_mask, cv2.MORPH_OPEN, kernel, iterations=3)

                x, y = obj['bitmap']['origin']
                _mask_height, _mask_width = _mask.shape
                mask[y: y + _mask_height, x:x + _mask_width] = _mask
            else:
                continue

    mask = np.expand_dims(mask, axis=2)
    return mask


if __name__ == '__main__':

    # The code snippet below is used only for debugging
    image_paths, ann_paths, dataset_names = read_supervisely_project(sly_project_dir='dataset/covid_segmentation_single_crop',
                                                                     included_datasets=[
                                                                         'MIMIC',
                                                                         'Vin-Dxr'
                                                                                        ])
    for idx in range(30):
        ann_path = ann_paths[idx]
        mask = convert_ann_to_mask(ann_path=ann_path,
                                   filter_mask=True)
