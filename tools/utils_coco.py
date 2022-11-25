import os
from typing import Dict, List, Tuple, Any

import cv2
import pandas as pd
from pathlib import Path

from tools.utils import get_file_list
from tools.utils_sly import ANNOTATION_COLUMNS


def get_img_info(
    img_path: str,
    img_id: int,
) -> Dict[str, Any]:
    img_data = {}
    height, width = cv2.imread(img_path).shape[:2]
    img_data['id'] = img_id                 # Unique image ID
    img_data['width'] = width
    img_data['height'] = height
    img_data['file_name'] = os.path.basename(img_path)
    return img_data


def get_ann_info(
    label_path: str,
    img_id: int,
    ann_id: int,
) -> Tuple[List[Any], int]:
    ann_data = []
    if os.path.exists(label_path):
        df_ann = pd.read_csv(label_path, sep='\t', names=ANNOTATION_COLUMNS)
        for _, row in df_ann.iterrows():
            label = {}
            x1, y1 = int(row['x1']), int(row['y1'])
            width = int(row['x2'] - row['x1'])
            height = int(row['y2'] - row['y1'])

            label['id'] = ann_id            # Should be unique
            label['image_id'] = img_id      # Image ID annotation relates to
            label['category_id'] = int(row['Class ID'])
            label['bbox'] = [x1, y1, width, height]
            label['area'] = width * height
            label['iscrowd'] = 0

            ann_data.append(label)
            ann_id += 1

    return ann_data, ann_id


def create_tn_subset(
    data_dir: str,
) -> Tuple[List[str], List[str]]:
    img_list = get_file_list(
        src_dirs=data_dir,
        include_template='',
        ext_list=[
            '.png',
            '.jpg',
            '.jpeg',
            '.bmp',
        ],
    )

    ann_list = []
    for img_path in img_list:
        ann_list.append(str(Path(img_path).with_suffix('.txt')))

    return img_list, ann_list


if __name__ == '__main__':
    a = get_img_info('dataset/MIMIC-CXR-Edema-Intermediate/img/10000980_54935705.png', 12)
    print(a)
    b, c = get_ann_info('dataset/MIMIC-CXR-Edema-Intermediate/ann/10000980_54935705.txt', 1, 2)
    print(c, b)
    x, y = create_tn_subset(data_dir='dataset/MIMIC-CXR-Edema-Intermediate')
    print(x, y)
