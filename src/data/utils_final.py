import logging
from typing import List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.utils_sly import FEATURE_MAP_REVERSED, get_box_sizes


def get_bboxes(
    df: pd.DataFrame,
) -> List[List[Union[int, float]]]:
    bboxes = []
    for _, row in df.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        bbox = [x1, y1, x2, y2]
        bboxes.append(bbox)
    return bboxes


def get_features(
    df: pd.DataFrame,
) -> List[str]:
    features = df['Feature'].tolist()
    return features


def set_bboxes(
    df: pd.DataFrame,
    bboxes: List[List[Union[int, float]]],
) -> pd.DataFrame:
    for i, bbox in enumerate(bboxes):
        df.loc[i, 'x1'] = bbox[0]
        df.loc[i, 'y1'] = bbox[1]
        df.loc[i, 'x2'] = bbox[2]
        df.loc[i, 'y2'] = bbox[3]
    return df


def set_features(
    df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    df['Feature'] = features
    return df


def update_image_metadata(
    df: pd.DataFrame,
    img_size: List[int],
) -> pd.DataFrame:
    img_height = img_size[0]
    img_width = img_size[1]
    img_ratio = img_height / img_width
    df['Image height'] = img_height
    df['Image width'] = img_width
    df['Image ratio'] = img_ratio
    return df


def update_bbox_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:
    for idx in df.index:
        box_sizes = get_box_sizes(
            x1=df.at[idx, 'x1'],
            y1=df.at[idx, 'y1'],
            x2=df.at[idx, 'x2'],
            y2=df.at[idx, 'y2'],
        )
        df.at[
            idx,
            ['xc', 'yc', 'Box width', 'Box height', 'Box ratio', 'Box area', 'Box label'],
        ] = box_sizes
    return df


def modify_box_geometry(
    df: pd.DataFrame,
    box_extension: dict,
) -> pd.DataFrame:
    for idx in tqdm(df.index, desc='Modify box geometry', unit=' boxes'):
        box_extension_feature = box_extension[FEATURE_MAP_REVERSED[df.at[idx, 'Feature ID']]]

        image_width = df.at[idx, 'Image width']
        image_height = df.at[idx, 'Image height']

        x1 = df.at[idx, 'x1'] - box_extension_feature[0]
        y1 = df.at[idx, 'y1'] - box_extension_feature[1]
        x2 = df.at[idx, 'x2'] + box_extension_feature[2]
        y2 = df.at[idx, 'y2'] + box_extension_feature[3]

        # Check if the box coordinates exceed image dimensions
        if x1 < 0:
            logging.warning(
                f'x1 = {x1} exceeds the left edge of the image = {0}. '
                f'Image: {df.at[idx, "Image name"]}. '
                f'Feature: {df.at[idx, "Feature"]}',
            )
        if y1 < 0:
            logging.warning(
                f'y1 = {y1} exceeds the top edge of the image = {0}. '
                f'Image: {df.at[idx, "Image name"]}. '
                f'Feature: {df.at[idx, "Feature"]}',
            )
        if x2 > image_width:
            logging.warning(
                f'x2 = {x2} exceeds the right edge of the image = {image_width}. '
                f'Image: {df.at[idx, "Image name"]}. '
                f'Feature: {df.at[idx, "Feature"]}',
            )
        if y2 > image_height:
            logging.warning(
                f'y2 = {y2} exceeds the bottom edge of the image = {image_height}. '
                f'Image: {df.at[idx, "Image name"]}. '
                f'Feature: {df.at[idx, "Feature"]}',
            )

        # Check if x2 is greater than x1 and y2 is greater than y1
        if x2 <= x1:
            logging.warning(
                f'x2 = {x2} is not greater than x1 = {x1}. '
                f'Image: {df.at[idx, "Image name"]}. '
                f'Feature: {df.at[idx, "Feature"]}',
            )
        if y2 <= y1:
            logging.warning(
                f'y2 = {y2} is not greater than y1 = {y1}. '
                f'Image: {df.at[idx, "Image name"]}. '
                f'Feature: {df.at[idx, "Feature"]}',
            )

        # Clip coordinates to image dimensions if necessary
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        # Update object coordinates and relative metadata
        df.at[idx, 'x1'] = x1
        df.at[idx, 'y1'] = y1
        df.at[idx, 'x2'] = x2
        df.at[idx, 'y2'] = y2

    logging.warning('All coordinates that exceed image dimensions are clipped')

    return df


def crop_image(
    img: np.ndarray,
    bboxes: List[List[Union[int, float]]],
    features: List[str],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Tuple[np.ndarray, List[List[Union[int, float]]], List[str]]:
    transform = A.Compose(
        [
            A.Crop(
                x_min=x1,
                y_min=y1,
                x_max=x2,
                y_max=y2,
                always_apply=True,
            ),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['class_labels'],
        ),
    )

    trans = transform(
        image=img,
        bboxes=bboxes,
        class_labels=features,
    )
    img_trans = trans['image']
    bboxes_trans = trans['bboxes']
    features_trans = trans['class_labels']

    return img_trans, bboxes_trans, features_trans


def resize_image(
    img: np.ndarray,
    bboxes: List[List[Union[int, float]]],
    features: List[str],
    output_size: List[int],
) -> Tuple[np.ndarray, List[List[Union[int, float]]], List[str]]:
    transform = A.Compose(
        [
            A.LongestMaxSize(
                max_size=max(output_size),
                interpolation=4,
                always_apply=True,
            ),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['class_labels'],
        ),
    )

    trans = transform(
        image=img,
        bboxes=bboxes,
        class_labels=features,
    )
    img_trans = trans['image']
    bboxes_trans = trans['bboxes']
    features_trans = trans['class_labels']

    return img_trans, bboxes_trans, features_trans


def pad_image(
    img: np.ndarray,
    bboxes: List[List[Union[int, float]]],
    features: List[str],
    output_size: List[int],
) -> Tuple[np.ndarray, List[List[Union[int, float]]], List[str]]:
    transform = A.Compose(
        [
            A.PadIfNeeded(
                min_width=output_size[0],
                min_height=output_size[1],
                position='center',
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                always_apply=True,
            ),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['class_labels'],
        ),
    )

    trans = transform(
        image=img,
        bboxes=bboxes,
        class_labels=features,
    )
    img_trans = trans['image']
    bboxes_trans = trans['bboxes']
    features_trans = trans['class_labels']

    return img_trans, bboxes_trans, features_trans
