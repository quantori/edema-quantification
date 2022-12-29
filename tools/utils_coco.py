import os
from typing import Dict, List, Tuple, Any

import cv2
import pandas as pd

from tools.utils_sly import ANNOTATION_COLUMNS, FIGURE_MAP_REVERSED


def get_img_info(
    img_path: str,
    img_id: int,
) -> Dict[str, Any]:
    img_data = {}
    height, width = cv2.imread(img_path).shape[:2]
    img_data['id'] = img_id  # Unique image ID
    img_data['width'] = width
    img_data['height'] = height
    img_data['file_name'] = os.path.basename(img_path)
    return img_data


def get_ann_info(
    label_path: str,
    img_id: int,
    ann_id: int,
    box_extension: dict,
) -> Tuple[List[Any], int]:
    ann_data = []
    if os.path.exists(label_path):
        df_ann = pd.read_csv(label_path, sep='\t', names=ANNOTATION_COLUMNS)
        for _, row in df_ann.iterrows():
            label = {}
            if row['Class ID'] > 0:
                box_extension_figure = box_extension[FIGURE_MAP_REVERSED[row['Figure ID']]]
                x1, y1 = (
                    int(row['x1']) - box_extension_figure[0],
                    int(row['y1']) - box_extension_figure[1],
                )
                x2, y2 = (
                    int(row['x2']) + box_extension_figure[0],
                    int(row['y2']) + box_extension_figure[1]
                )
                width = abs(x2 - x1 + 1)
                height = abs(y2 - y1 + 1)

                label['id'] = ann_id            # Should be unique
                label['image_id'] = img_id      # Image ID annotation relates to
                label['category_id'] = int(row['Figure ID'])
                label['bbox'] = [x1, y1, width, height]
                label['area'] = width * height
                label['iscrowd'] = 0

                ann_data.append(label)
                ann_id += 1
            else:
                return [], 0

    return ann_data, ann_id
