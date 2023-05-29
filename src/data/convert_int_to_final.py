import logging
import os
from pathlib import Path
from typing import List

import albumentations as A
import cv2
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils_sly import FEATURE_MAP_REVERSED, get_box_sizes

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_bboxes(
    df: pd.DataFrame,
) -> List[List[int]]:
    bboxes = []
    for _, row in df.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        bbox = [x1, y1, x2, y2]
        bboxes.append(bbox)
    return bboxes


def get_features(
    df: pd.DataFrame,
) -> List[str]:
    df_out = df.copy(deep=True)
    features = df_out['Feature'].tolist()
    return features


def set_bboxes(
    df: pd.DataFrame,
    bboxes: List[List[int]],
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
            log.warning(
                f'x1 = {x1} is out of bound = {0}. Image: {df.at[idx, "Image name"]}, Feature: {df.at[idx, "Feature"]}',
            )
        if y1 < 0:
            log.warning(
                f'y1 = {y1} is out of bound = {0}. Image: {df.at[idx, "Image name"]}, Feature: {df.at[idx, "Feature"]}',
            )
        if x2 > image_width:
            log.warning(
                f'x2 = {x2} is out of bound = {image_width}. Image: {df.at[idx, "Image name"]}, Feature: {df.at[idx, "Feature"]}',
            )
        if y2 > image_height:
            log.warning(
                f'y2 = {y2} is out of bound = {image_height}. Image: {df.at[idx, "Image name"]}, Feature: {df.at[idx, "Feature"]}',
            )

        # Check if x2 is greater than x1 and y2 is greater than y1
        if x2 <= x1:
            log.warning(
                f'x2 = {x2} is not greater than x1 = {x1}. Image: {df.at[idx, "Image name"]}, Feature: {df.at[idx, "Feature"]}',
            )
        if y2 <= y1:
            log.warning(
                f'y2 = {y2} is not greater than y1 = {y1}. Image: {df.at[idx, "Image name"]}, Feature: {df.at[idx, "Feature"]}',
            )

        # Clamp coordinates to image dimensions if necessary
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        # Update object coordinates and relative metadata
        df.at[idx, 'x1'] = x1
        df.at[idx, 'y1'] = y1
        df.at[idx, 'x2'] = x2
        df.at[idx, 'y2'] = y2

    return df


def crop_images(
    df: pd.DataFrame,
    output_size: List[int],
    save_dir: str,
) -> pd.DataFrame:
    # Process boxes for each image independently
    df_out = pd.DataFrame(columns=df.columns)
    img_dir = os.path.join(save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    gb = df.groupby(['Image path'])

    # TODO: Fix view for 10263098_52746676.png Effusion (Lateral exceeding)
    # TODO: Fix view for 12152816_58885266.png Effusion (Frontal exceeding)
    for img_path, df_img in tqdm(gb, desc='Processing images..', unit=' images'):
        df_img.reset_index(drop=True, inplace=True)
        df_lungs = df_img.loc[df_img['Feature'] == 'Lungs']
        x1 = df_lungs.at[df_lungs.index[0], 'x1']
        y1 = df_lungs.at[df_lungs.index[0], 'y1']
        x2 = df_lungs.at[df_lungs.index[0], 'x2']
        y2 = df_lungs.at[df_lungs.index[0], 'y2']

        # Define transformation
        transform = A.Compose(
            [
                A.Crop(
                    x_min=x1,
                    y_min=y1,
                    x_max=x2,
                    y_max=y2,
                    always_apply=True,
                ),
                A.LongestMaxSize(
                    max_size=max(output_size),
                    interpolation=1,
                    always_apply=True,
                ),
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

        # Crop image and update metadata
        img = cv2.imread(img_path)
        bboxes = get_bboxes(df_img)
        features = get_features(df_img)
        trans = transform(
            image=img,
            bboxes=bboxes,
            class_labels=features,
        )
        img_trans = trans['image']
        bboxes_trans = [[round(value) for value in tup] for tup in trans['bboxes']]
        features_trans = trans['class_labels']
        if len(bboxes) != len(bboxes_trans):
            log.warn(f'Different number of boxes after transformation: {Path(img_path).name}')

        # Update box and image metadata
        df_img = set_bboxes(
            df=df_img,
            bboxes=bboxes_trans,
        )
        df_img = set_features(
            df=df_img,
            features=features_trans,
        )
        df_img = update_image_metadata(
            df=df_img,
            img_size=img_trans.shape[:2],
        )
        df_img = update_bbox_metadata(
            df=df_img,
        )

        # Save image and update corresponding dataframe
        img_name = Path(img_path).name
        save_path = os.path.join(img_dir, img_name)
        cv2.imwrite(save_path, img_trans)
        df_out = pd.concat([df_out, df_img], axis=0)
    df_out.sort_values(by=['Image path'], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    return df


def _merge_metadata(
    df1_path: str,
    df2_path: str,
) -> pd.DataFrame:
    # Read the metadata
    df1 = pd.read_excel(df1_path)
    df2 = pd.read_excel(df2_path)

    # Merge the data frames
    df_merged = pd.concat([df1, df2], axis=0)
    df_merged.drop(columns=['ID'], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    # Fill empty fields
    df_out = pd.DataFrame(columns=df_merged.columns)
    gb = df_merged.groupby(['Image name', 'Subject ID', 'Study ID'])
    columns_to_fill = ['Image path', 'Dataset', 'Class', 'Class ID']
    for _, df_sample in gb:
        for column in columns_to_fill:
            unique_values = df_sample[column].dropna().unique()
            assert len(unique_values) == 1, f'Column "{column}" has more than one unique value'
            df_sample[column].fillna(unique_values[0], inplace=True)
        df_out = pd.concat([df_out, df_sample], axis=0)
    df_out.sort_values(by=['Image path'], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    return df_out


def process_metadata(
    dataset_dir: str,
    dataset_dir_fused: str,
) -> pd.DataFrame:
    """Process dataset metadata.

    Args:
        dataset_dir: a path to the directory containing series with images and labels
        dataset_dir_fused: path to a directory containing fused probability maps and their metadata
    Returns:
        metadata: an updated metadata dataframe
    """
    metadata = _merge_metadata(
        df1_path=os.path.join(dataset_dir, 'metadata.xlsx'),
        df2_path=os.path.join(dataset_dir_fused, 'metadata.xlsx'),
    )
    metadata = metadata[metadata['View'] == 'Frontal']
    metadata = metadata.dropna(subset=['Class ID'])
    metadata = metadata.drop(['Mask', 'Points'], axis=1)

    return metadata


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_int_to_final',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    """Convert intermediate dataset to COCO.

    Args:
        dataset_dir: path to directory containing series with images and labels inside
        dataset_dir_fused: path to a directory containing fused probability maps and their metadata
        output_size: a target output size for the images
        box_extension: a value used to extend or contract object box sizes
        save_dir: directory where split datasets are saved to
    Returns:
        None
    """
    log.info(f'Source data directory.....: {cfg.dataset_dir}')
    log.info(f'Fused data directory......: {cfg.dataset_dir_fused}')
    log.info(f'Output size...............: {cfg.output_size}')
    log.info(f'Box extension.............: {cfg.box_extension}')
    log.info(f'Output directory..........: {cfg.save_dir}')

    # Process source metadata
    metadata = process_metadata(
        dataset_dir=cfg.dataset_dir,
        dataset_dir_fused=cfg.dataset_dir_fused,
    )

    # Update box coordinates with box_extension
    metadata = modify_box_geometry(
        df=metadata,
        box_extension=cfg.box_extension,
    )

    # Crop and save images
    metadata = crop_images(
        df=metadata,
        output_size=cfg.output_size,  # TODO: check
        save_dir=cfg.save_dir,
    )

    # Save updated metadata
    metadata['Confidence'] = 1.0
    metadata.reset_index(drop=True, inplace=True)
    save_path = os.path.join(cfg.save_dir, 'metadata.xlsx')
    metadata.index += 1
    metadata.to_excel(
        save_path,
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
