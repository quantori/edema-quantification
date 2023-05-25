import base64
import logging
import os
import zlib
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import supervisely_lib as sly

CLASS_MAP = {
    '': None,
    'No edema': 0,
    'Vascular congestion': 1,
    'Interstitial edema': 2,
    'Alveolar edema': 3,
}

FEATURE_MAP = {
    'Cephalization': 1,
    'Heart': 2,
    'Artery': 3,
    'Bronchus': 4,
    'Kerley': 5,
    'Cuffing': 6,
    'Effusion': 7,
    'Bat': 8,
    'Infiltrate': 9,
    'Lungs': 10,
}

FEATURE_MAP_REVERSED = dict((v, k) for k, v in FEATURE_MAP.items())

FEATURE_TYPE = {
    'Cephalization': 'line',
    'Artery': 'bitmap',
    'Heart': 'rectangle',
    'Kerley': 'line',
    'Bronchus': 'bitmap',
    'Effusion': 'polygon',
    'Bat': 'polygon',
    'Infiltrate': 'polygon',
    'Cuffing': 'bitmap',
    'Lungs': 'bitmap',
}

METADATA_COLUMNS = [
    'Image path',
    'Image name',
    'Subject ID',
    'Study ID',
    'Dataset',
    'Image width',
    'Image height',
    'Image ratio',
    'Feature ID',
    'Feature',
    'Source type',
    'Reference type',
    'Match',
    'x1',
    'y1',
    'x2',
    'y2',
    'xc',
    'yc',
    'Box width',
    'Box height',
    'Box ratio',
    'Box area',
    'Box label',
    'RP',
    'Mask',
    'Points',
    'View',
    'Class ID',
    'Class',
]


def read_sly_project(
    dataset_dir: str,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read the Supervisely project as a dataframe.

    Args:
        dataset_dir: a path to Supervisely dataset directory
        include_dirs: a list of subsets to include in the dataset
        exclude_dirs: a list of subsets to exclude from the dataset
    Returns:
        df: dataframe representing the dataset
    """
    logging.info(f'Dataset dir..........: {dataset_dir}')
    assert os.path.exists(dataset_dir) and os.path.isdir(
        dataset_dir,
    ), 'Wrong project dir: {}'.format(dataset_dir)
    project = sly.Project(
        directory=dataset_dir,
        mode=sly.OpenMode.READ,
    )

    img_paths: List[str] = []
    ann_paths: List[str] = []
    subset_list: List[str] = []

    for dataset in project:
        subset = dataset.name

        if include_dirs and subset not in include_dirs:
            logging.info(f'Excluded dir.........: {subset}')
            continue

        if exclude_dirs and subset in exclude_dirs:
            logging.info(f'Excluded dir.........: {subset}')
            continue

        logging.info(f'Included dir.........: {subset}')
        for item_name in dataset:
            img_path, ann_path = dataset.get_item_paths(item_name)
            img_paths.append(img_path)
            ann_paths.append(ann_path)
            subset_list.append(subset)

    df = pd.DataFrame.from_dict(
        {
            'img_path': img_paths,
            'ann_path': ann_paths,
            'subset': subset_list,
        },
    )
    df.sort_values(['subset'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def convert_base64_to_image(
    encoded_mask: str,
) -> np.ndarray:
    """The convert_base64_to_image function converts a base64 encoded string to a numpy array.

    Args:
        encoded_mask: bitmap represented as a string
    Returns:
        mask: bitmap represented as a numpy array
    """
    z = zlib.decompress(base64.b64decode(encoded_mask))
    n = np.frombuffer(z, np.uint8)

    img_decoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if (len(img_decoded.shape) == 3) and (img_decoded.shape[2] >= 4):
        mask = img_decoded[:, :, 3].astype(np.uint8)
    elif len(img_decoded.shape) == 2:
        mask = img_decoded.astype(np.uint8)
    else:
        raise RuntimeError('Wrong internal mask format.')
    return mask


def get_class_name(
    ann: dict,
) -> str:
    """Extract a class name from an annotation.

    Args:
        ann: a dictionary with Supervisely annotations
    Returns:
        class_name: name of the class for a given image
    """
    if ann['tags']:
        class_name = ann['tags'][0]['value']
    else:
        # logging.warning(f'No class tags available {ann}')
        class_name = ''
    return class_name


def get_tag_value(
    obj: dict,
    tag_name: str,
) -> str:
    """Extract a tag value from an annotation.

    Args:
        tag_name: tag name for searching
        obj: dictionary with information about one object from supervisely annotations
    Returns:
        tag_value: string with value of tag name
    """
    if obj['tags']:
        tag_value_list = [v['value'] for v in obj['tags'] if v['name'] == tag_name]
        if tag_value_list:
            tag_value = tag_value_list[0]
        else:
            # logging.warning(f'No {tag_name} value in {obj}')
            tag_value = ''
    else:
        # logging.warning(f'No {tag_name} value in {obj}')
        tag_value = ''
    return tag_value


def get_mask_points(
    obj: dict,
) -> dict:
    """Extract a mask and a list of points from a Supervisely annotation.

    Args:
        obj: dictionary with information about one object from Supervisely annotations
    Returns:

    """
    if obj['geometryType'] == 'bitmap':
        return {
            'Mask': obj['bitmap']['data'],
            'Points': [int(np.round(s)) for s in obj['bitmap']['origin']],
        }
    else:
        return {
            'Mask': None,
            'Points': [[int(np.round(s)) for s in lst] for lst in obj['points']['exterior']],
        }


def get_object_box(
    obj: dict,
) -> dict:
    """Extract box coordinates from a Supervisely annotation.

    Args:
        obj: dictionary with information about one object from supervisely annotations
    Returns:
        dictionary which contains coordinates for a rectangle (left, top, right, bottom)
    """
    if obj['geometryType'] == 'bitmap':
        bitmap = convert_base64_to_image(obj['bitmap']['data'])
        x1, y1 = obj['bitmap']['origin'][0], obj['bitmap']['origin'][1]
        x2 = x1 + bitmap.shape[1]
        y2 = y1 + bitmap.shape[0]
    else:
        xs = [x[0] for x in obj['points']['exterior']]
        ys = [x[1] for x in obj['points']['exterior']]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    return {
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
    }


def get_box_sizes(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> dict:
    """Extract box sizes by its coordinates.

    Args:
        x1: left x
        y1: top y
        x2: right x
        y2: bottom y
    Returns:
        dictionary which contains coordinates for rectangle (a center point and a width/height)
    """
    box_width = abs(x2 - x1 + 1)
    box_height = abs(y2 - y1 + 1)
    xc = x1 + box_width // 2
    yc = y1 + box_height // 2
    box_area = box_height * box_width
    box_ratio = box_height / box_width
    if box_area < 32 * 32:
        box_label = 'Small'
    elif 32 * 32 <= box_area <= 96 * 96:
        box_label = 'Medium'
    else:
        box_label = 'Large'

    return {
        'xc': xc,
        'yc': yc,
        'Box width': box_width,
        'Box height': box_height,
        'Box ratio': box_ratio,
        'Box area': box_area,
        'Box label': box_label,
    }
