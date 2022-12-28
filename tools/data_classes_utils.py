import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

import torch
from torch.utils.data import Dataset, Subset

import cv2
import albumentations as A
from PIL import Image, ImageDraw, ImageOps

from tools.utils_sly import FIGURE_MAP, convert_base64_to_image


# all relevant edema findings for which masks will be prepared subsequently
EDEMA_FINDINGS = ['no_findings'] + [k for k in FIGURE_MAP.keys() if k != 'Heart']


def parse_coord_string(coord_string: str) -> np.ndarray:
    coordinates = [int(d) for d in re.findall(r'\d+', coord_string)]
    return np.array(coordinates).reshape(-1, 2)


def extract_annotations(group_df: pd.DataFrame) -> Dict[str, defaultdict]:

    annotations = {k: defaultdict(list) for k in group_df['Figure'].unique()}

    for _, data in group_df[['Figure', 'x1', 'y1', 'Mask', 'Points']].iterrows():

        if pd.notna(data['Mask']):
            annotations[data['Figure']]['bitmaps'].append(data.loc['x1': 'Mask'].to_dict())
        else:
            annotations[data['Figure']]['polygons'].append(parse_coord_string(data['Points']))

    return annotations


def make_masks(image: Image.Image,
               annotations: Dict,
               linelike_finding_width: int = 15,
               ) -> Tuple[List[np.ndarray], List[int], int]:
    """

    Args:
        image: image
        annotations: dict with annotation data for a given image
        linelike_finding_width: line width in masks corresponding to findings annotated with lines

    Returns:
        masks: list of masks represented as numpy arrays for each edema finding
        findings: list of 0/1 values showing the absence/presence of a given finding in a given image
        default_mask_value: value with which mask is padded depending on the
                            presence of at least one of the edema findings in the image
    """

    masks = []
    findings = []
    width, height = image.size
    default_mask_value = 1 if any(f in EDEMA_FINDINGS for f in annotations.keys()) else 0

    for finding in EDEMA_FINDINGS:
        # binary mask template
        finding_mask = Image.new(mode='1', size=(width, height), color=default_mask_value)

        if finding in annotations.keys():

            # draw finding mask represented by polygons
            if annotations[finding]['polygons']:

                draw = ImageDraw.Draw(finding_mask)

                for point_array in annotations[finding]['polygons']:
                    # for line-like findings we draw lines with default width of 15px
                    if finding in ('Kerley', 'Cephalization'):
                        draw.line(point_array.flatten().tolist(), fill=0,
                                  width=linelike_finding_width)
                    else:
                        draw.polygon(point_array.flatten().tolist(), fill=0, outline=0)

            # draw finding mask represented by bitmaps
            elif annotations[finding]['bitmaps']:

                for bitmap in annotations[finding]['bitmaps']:
                    bitmap_array = convert_base64_to_image(bitmap['Mask'])
                    bitmap_mask = Image.fromarray(bitmap_array).convert('1')
                    inverted_bitmap_mask = ImageOps.invert(bitmap_mask)

                    finding_mask.paste(inverted_bitmap_mask, box=(bitmap['x1'], bitmap['y1']))

            else:
                raise RuntimeError('neither polygon nor mask data is present in the metadata')

            masks.append(np.array(finding_mask, dtype=np.uint8))
            findings.append(1)

        else:
            masks.append(np.array(finding_mask, dtype=np.uint8))
            findings.append(0)

    return masks, findings, default_mask_value


def resize_and_create_masks(image: Image.Image,
                            annotations: Dict,
                            target_size: Tuple[int, int],
                            linelike_finding_width: int = 15,
                            ) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:

    """

    Args:
        image: input image
        annotations: dict with image annotations
        target_size: tuple with target image size (width, height)
        linelike_finding_width: width in pixels for linelike annotations

    Returns:
        image_resized: numpy array of the resized image
        masks_resized: list of numpy arrays of the resized masks
        findings: list of 0/1 values showing the presence/absence of a given finding in a given image
    """

    # image and created image masks with initial size
    image_arr = np.array(image)
    masks, findings, default_mask_value = make_masks(
        image, annotations, linelike_finding_width=linelike_finding_width
    )

    # further we resize image and masks by padding them while keeping initial aspect ratio
    # this transform mimics PIL.ImageOps.pad function
    # therefore we define resize_transform depending on several factors
    width_new, height_new = target_size
    width_0, height_0 = image.size
    aspect_0 = width_0 / height_0
    aspect_new = width_new / height_new

    if aspect_0 < aspect_new:
        if width_0 > height_0:
            resize_transform = A.augmentations.geometric.SmallestMaxSize(max_size=height_new)
        else:
            resize_transform = A.augmentations.geometric.LongestMaxSize(max_size=height_new)
    elif aspect_0 > aspect_new:
        if width_0 > height_0:
            resize_transform = A.augmentations.geometric.LongestMaxSize(max_size=width_new)
        else:
            resize_transform = A.augmentations.geometric.SmallestMaxSize(max_size=width_new)
    else:
        resize_transform = A.augmentations.geometric.Resize(height_new, width_new)

    # make image and masks resize operations
    # image is padded with 0 and masks with default_mask_value (0/1)
    transform = A.Compose([
        resize_transform,
        A.augmentations.geometric.PadIfNeeded(height_new, width_new,
                                              border_mode=cv2.BORDER_CONSTANT,
                                              value=0, mask_value=default_mask_value)
    ])

    transformed = transform(image=image_arr, masks=masks)
    image_resized = transformed['image']
    masks_resized = transformed['masks']

    return image_resized, masks_resized, findings


def combine_image_and_masks(transformed_image: torch.Tensor,
                            transformed_masks: List[torch.Tensor],
                            ) -> torch.Tensor:
    transformed_image = [transformed_image]
    expanded_masks = [m.expand(1, -1, -1) for m in transformed_masks]
    tensor = torch.cat(transformed_image + expanded_masks, dim=0)

    return tensor


def split_dataset(dataset: Dataset,
                  train_share: float,
                  metadata_df: pd.DataFrame,
                  show_class_distributions: bool = False,
                  verbose: bool = False,
                  ) -> Tuple[Subset, Subset]:

    # split dataset using: stratification, groups on subject ID, default ratio is 80:20
    test_share = 1 - train_share
    n_splits = int(train_share / test_share) + 1
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    X = metadata_df['Image path'].values
    y = metadata_df['Class ID'].astype(int).values
    groups = metadata_df['Subject ID'].values

    rmsd_min = np.inf
    best_train_indices, best_test_indices = None, None

    for i, (train_index, test_index) in enumerate(sgkf.split(X, y, groups)):

        dfc = metadata_df.copy()
        dfc.loc[dfc.index.isin(train_index), 'train_test_set'] = 'train'
        dfc.loc[dfc.index.isin(test_index), 'train_test_set'] = 'test'

        df_percentages = dfc.groupby('train_test_set')['Class ID']\
                            .value_counts(normalize=True)\
                            .unstack()

        rmsd_classes = np.sum((df_percentages.loc['train'] - df_percentages.loc['test']) ** 2)
        if rmsd_classes < rmsd_min:
            rmsd_min = rmsd_classes
            best_train_indices, best_test_indices = train_index, test_index

        if verbose:
            print(f'Fold {i}:')
            print(f'  Train size: {train_index.size}')
            print(f'  Test  size: {test_index.size}')
            print(f'  rmsd for classes distribution in train/test split: {rmsd_classes:.5f}')

        if show_class_distributions:
            df_percentages.plot.barh(figsize=(6, 3))
            plt.show()

    train_subset = Subset(dataset, best_train_indices)
    test_subset = Subset(dataset, best_test_indices)

    return train_subset, test_subset
