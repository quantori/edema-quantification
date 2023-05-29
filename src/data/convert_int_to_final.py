import logging
import os
from pathlib import Path
from typing import List

import cv2
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils_final import (
    crop_image,
    get_bboxes,
    get_features,
    modify_box_geometry,
    pad_image,
    resize_image,
    set_bboxes,
    set_features,
    update_bbox_metadata,
    update_image_metadata,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def process_images(
    df: pd.DataFrame,
    enable_cropping: bool,
    enable_resizing: bool,
    enable_padding: bool,
    output_size: List[int],
    save_dir: str,
) -> pd.DataFrame:
    # Process images independently
    df_out = pd.DataFrame(columns=df.columns)
    img_dir = os.path.join(save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    gb = df.groupby(['Image path'])

    for img_path, df_img in tqdm(gb, desc='Processing images', unit=' images'):
        # Read image and get box dimensions
        df_img.reset_index(drop=True, inplace=True)
        img = cv2.imread(img_path)
        bboxes = get_bboxes(df_img)
        features = get_features(df_img)

        # Crop using lung coordinates
        if enable_cropping:
            df_lungs = df_img.loc[df_img['Feature'] == 'Lungs']
            assert len(df_lungs) == 1, 'More than one lung object found'
            x1 = df_lungs.at[df_lungs.index[0], 'x1']
            y1 = df_lungs.at[df_lungs.index[0], 'y1']
            x2 = df_lungs.at[df_lungs.index[0], 'x2']
            y2 = df_lungs.at[df_lungs.index[0], 'y2']

            img, bboxes, features = crop_image(
                img=img,
                bboxes=bboxes,
                features=features,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )

        # Resize image while keeping aspect ratio
        if enable_resizing:
            img, bboxes, features = resize_image(
                img=img,
                bboxes=bboxes,
                features=features,
                output_size=output_size,
            )

        # Pad image
        if enable_padding:
            img, bboxes, features = pad_image(
                img=img,
                bboxes=bboxes,
                features=features,
                output_size=output_size,
            )

        # Round box coordinates
        bboxes = [[round(value) for value in tup] for tup in bboxes]

        # Update box and image metadata
        df_img = set_bboxes(
            df=df_img,
            bboxes=bboxes,
        )
        df_img = set_features(
            df=df_img,
            features=features,
        )
        df_img = update_bbox_metadata(
            df=df_img,
        )
        df_img = update_image_metadata(
            df=df_img,
            img_size=img.shape[:2],
        )

        # Save image and update corresponding dataframe
        img_name = Path(img_path).name
        save_path = os.path.join(img_dir, img_name)
        cv2.imwrite(save_path, img)
        df_out = pd.concat([df_out, df_img], axis=0)
    df_out.sort_values(by=['Image path'], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    return df_out


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
    df = _merge_metadata(
        df1_path=os.path.join(dataset_dir, 'metadata.xlsx'),
        df2_path=os.path.join(dataset_dir_fused, 'metadata.xlsx'),
    )
    df = df[df['View'] == 'Frontal']
    df = df.dropna(subset=['Class ID'])
    df = df.drop(['Mask', 'Points'], axis=1)

    return df


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
    log.info(f'Enable cropping...........: {cfg.enable_cropping}')
    log.info(f'Enable resizing...........: {cfg.enable_resizing}')
    log.info(f'Enable padding............: {cfg.enable_padding}')
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

    # Process and save images
    metadata = process_images(
        df=metadata,
        enable_cropping=cfg.enable_cropping,
        enable_resizing=cfg.enable_resizing,
        enable_padding=cfg.enable_padding,
        output_size=cfg.output_size,
        save_dir=cfg.save_dir,
    )

    # Save updated metadata
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
