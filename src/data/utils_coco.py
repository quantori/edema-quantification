import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import pandas as pd


def get_img_info(
    img_path: str,
    img_id: int,
) -> Dict[str, Any]:
    img_data: Dict[str, Union[int, str]] = {}
    height, width = cv2.imread(img_path).shape[:2]
    img_data['id'] = img_id  # Unique image ID
    img_data['width'] = width
    img_data['height'] = height
    img_data['file_name'] = os.path.basename(img_path)
    return img_data


def get_ann_info(
    df: pd.DataFrame,
    img_id: int,
    ann_id: int,
) -> Tuple[List[Any], int]:
    ann_data = []
    for _, row in df.iterrows():
        label: Dict[str, Union[int, List[int]]] = {}
        if row['Class ID'] > 0:
            x1, y1 = (
                int(row['x1']),
                int(row['y1']),
            )
            x2, y2 = (
                int(row['x2']),
                int(row['y2']),
            )
            width = abs(x2 - x1 + 1)
            height = abs(y2 - y1 + 1)

            label['id'] = ann_id  # Should be unique
            label['image_id'] = img_id  # Image ID annotation relates to
            label['category_id'] = int(row['Feature ID'])
            label['bbox'] = [x1, y1, width, height]
            label['area'] = width * height
            label['iscrowd'] = 0

            ann_data.append(label)
            ann_id += 1
        else:
            return [], ann_id

    return ann_data, ann_id
