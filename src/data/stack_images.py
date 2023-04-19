import logging
import os
from functools import partial
from typing import Tuple

import cv2
import hydra
import imutils
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def stack_single_study(
    group: Tuple[int, pd.DataFrame],
    img_height: int,
    save_dir: str,
) -> pd.DataFrame:
    _, df_study = group
    df_study.reset_index(drop=True, inplace=True)

    # Extract paths of the frontal and lateral images
    df_frontal = df_study[(df_study['View'] == 'AP') | (df_study['View'] == 'PA')]
    df_lateral = df_study[(df_study['View'] == 'LL') | (df_study['View'] == 'LATERAL')]
    img_path_frontal = df_frontal.iloc[0]['Image path']
    img_path_lateral = df_lateral.iloc[0]['Image path']

    # Read and resize images
    img_frontal = cv2.imread(img_path_frontal)
    img_lateral = cv2.imread(img_path_lateral)
    img_frontal = imutils.resize(img_frontal, height=img_height, inter=cv2.INTER_LINEAR)
    img_lateral = imutils.resize(img_lateral, height=img_height, inter=cv2.INTER_LINEAR)

    # Stack frontal and lateral images
    img_out = np.zeros([img_height, 1, 3], dtype=np.uint8)
    img_out = np.hstack([img_frontal, img_lateral, img_out])
    img_out = np.delete(img_out, 0, 1)

    # Save images
    subject_name = df_study.loc[0, 'Subject ID']
    study_name = df_study.loc[0, 'Study ID']
    img_name = f'{subject_name}_{study_name}_{img_frontal.shape[1]}_{img_lateral.shape[1]}.png'
    save_path = os.path.join(save_dir, 'files', img_name)
    cv2.imwrite(save_path, img_out)

    # Change study metadata
    df_out = df_frontal.copy(deep=True)
    df_out.reset_index(drop=True, inplace=True)

    # Update DICOM IDs
    dicom_id_frontal = df_frontal.iloc[0]['DICOM ID']
    dicom_id_lateral = df_lateral.iloc[0]['DICOM ID']
    dicom_id_idx = df_out.columns.get_loc('DICOM ID')
    df_out.insert(dicom_id_idx + 1, 'DICOM ID LL', dicom_id_lateral)
    df_out.at[0, 'DICOM ID'] = dicom_id_frontal
    df_out.rename(columns={'DICOM ID': 'DICOM ID FR'}, inplace=True)

    # Update view values
    df_out.at[0, 'View'] = 'STACKED'
    df_out.at[0, 'View Code'] = 'stacked'

    # Update image dimensions
    df_out.at[0, 'Height'] = img_out.shape[0]
    df_out.at[0, 'Width'] = img_out.shape[1]
    width_idx = df_out.columns.get_loc('Width')
    df_out.insert(width_idx + 1, 'Width FR', img_frontal.shape[1])
    df_out.insert(width_idx + 2, 'Width LL', img_lateral.shape[1])
    df_out.at[0, 'Image path'] = os.path.relpath(save_path, start=save_dir)
    df_out.at[0, 'Image name'] = img_name
    return df_out


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='stack_images',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    """Stack frontal and lateral images and save them.

    Args:
        dataset_dir: a path to the data directory with filtered images
        exclude_devices: exclude images with support devices
        img_height: the height of the output images
        save_dir: a path to directory where the output files will be saved
    Returns:
        None
    """
    # Read dataset metadata
    metadata_path = os.path.join(cfg.dataset_dir, 'metadata.csv')
    df = pd.read_csv(metadata_path)
    if cfg.exclude_devices:
        df = df[df['Support Devices'] != 1]
    df['Image path'] = df.apply(
        func=lambda row: os.path.join(cfg.dataset_dir, str(row['Image path'])),
        axis=1,
    )
    groups = df.groupby(['Study ID'])

    # Multiprocessing of the dataset
    img_dir = os.path.join(cfg.save_dir, 'files')
    os.makedirs(img_dir, exist_ok=True)
    processing_func = partial(
        stack_single_study,
        img_height=cfg.img_height,
        save_dir=cfg.save_dir,
    )
    result = Parallel(n_jobs=-1)(
        delayed(processing_func)(group)
        for group in tqdm(groups, desc='Stacking images', unit=' study')
    )
    df_out = pd.concat(result)
    df_out.reset_index(drop=True, inplace=True)
    df_out.sort_values(['Subject ID', 'ID'], inplace=True)
    save_path = os.path.join(cfg.save_dir, f'metadata.csv')
    df_out.to_csv(
        save_path,
        index=False,
    )


if __name__ == '__main__':
    main()
