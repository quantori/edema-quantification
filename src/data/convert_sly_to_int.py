import logging
import os
from functools import partial
from pathlib import Path

import cv2
import hydra
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.utils_sly import (
    CLASS_MAP,
    FEATURE_MAP,
    FEATURE_TYPE,
    METADATA_COLUMNS,
    get_box_sizes,
    get_class_name,
    get_mask_points,
    get_object_box,
    get_tag_value,
    read_sly_project,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def process_image(
    row: pd.Series,
    save_dir: str,
) -> dict:
    """Process a single image.

    Args:
        row: series with information about one image
        save_dir: directory where the output frontal images and metadata will be stored
    Returns:
        dictionary with information about one image
    """
    (img_path, ann_path), row = row
    log.info(f'Cropping image {img_path}')

    img_stem = Path(img_path).stem
    subject_id, study_id, width_frontal, width_lateral = img_stem.split('_')
    dataset = Path(img_path).parts[-3]
    img = cv2.imread(img_path)
    img_height = img.shape[0]
    img_width = int(width_frontal)
    img = img[0:img_height, 0:img_width]
    img_ratio = img_height / img_width

    save_dir_img = os.path.join(save_dir, 'img')
    os.makedirs(save_dir_img, exist_ok=True)
    img_path = os.path.join(save_dir_img, f'{subject_id}_{study_id}.png')
    cv2.imwrite(img_path, img)

    return {
        'Image path': img_path,
        'Image name': Path(img_path).name,
        'Subject ID': subject_id,
        'Study ID': study_id,
        'Dataset': dataset,
        'Image width': img_width,
        'Image height': img_height,
        'Image ratio': img_ratio,
    }


def process_annotation(
    row: pd.Series,
    img_info: dict,
) -> pd.DataFrame:
    """Process a single annotation.

    Args:
        row: series with information about one image
        img_info: dictionary with information about one image
    Returns:
        meta: dataframe with metadata for one image
    """
    log.info('Preparing metadata')
    (img_path, ann_path), row = row
    meta = pd.DataFrame(columns=METADATA_COLUMNS)

    ann = sly.io.json.load_json_file(ann_path)
    class_name = get_class_name(ann)

    for obj in ann['objects']:
        log.debug(f'Processing object {obj}')

        feature_name = obj['classTitle']
        rp = get_tag_value(obj, tag_name='RP')
        xy = get_object_box(obj)
        box = get_box_sizes(*xy.values())
        mask_points = get_mask_points(obj)
        if xy['x1'] > img_info['Image width'] or xy['x2'] > img_info['Image width']:
            view = 'Lateral'
        else:
            view = 'Frontal'

        obj_info = {
            'Feature ID': FEATURE_MAP[feature_name],
            'Feature': feature_name,
            'Source type': obj['geometryType'],
            'Reference type': FEATURE_TYPE[feature_name],
            'Match': int(obj['geometryType'] == FEATURE_TYPE[feature_name]),
            'RP': rp,
            'View': view,
            'Class ID': CLASS_MAP[class_name],
            'Class': class_name,
        }
        obj_info.update(img_info)
        obj_info.update(xy)
        obj_info.update(box)
        obj_info.update(mask_points)
        meta = meta.append(obj_info, ignore_index=True)

    return meta


def process_sample(
    row: pd.Series,
    save_dir: str,
) -> pd.DataFrame:
    """Process a single sample.

    Args:
        row: series with information about one image
        save_dir: directory where the output frontal images and metadata will be stored
    Returns:
        ann_info: dataframe with metadata for one image
    """
    img_info = process_image(row, save_dir)
    ann_info = process_annotation(row, img_info)

    return ann_info


def split_dataset(
    df: pd.DataFrame,
    train_size: float,
    seed: int,
) -> pd.DataFrame:
    """Split dataset with stratification into training and test subsets.

    Args:
        df: data frame derived from supervisely annotation JSON files
        train_size: a fraction used to split dataset into train and test subsets
        seed: random value for splitting train and test subsets
    Returns:
        df_out: data frame with train and test subsets
    """
    # Extract a subset of unique subject IDs with stratification
    df_unique_subjects = df.groupby(by='Subject ID', as_index=False)['Class ID'].max()

    # Split dataset into train and test subsets with
    train_ids, test_ids = train_test_split(
        df_unique_subjects,
        train_size=train_size,
        shuffle=True,
        random_state=seed,
        stratify=df_unique_subjects['Class ID'],
    )

    # Extract train and test subsets by indices
    df_train = df[df['Subject ID'].isin(train_ids['Subject ID'])]
    df_test = df[df['Subject ID'].isin(test_ids['Subject ID'])]
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    # Move cases without edema from test subset to training subset
    mask_empty = df_train['Class ID'] == 0
    df_empty = df_train[mask_empty]
    df_train = df_train.drop(df_train.index[mask_empty])
    df_test = df_test.append(df_empty, ignore_index=True)

    # Add split column
    df_train['Split'] = 'train'
    df_test['Split'] = 'test'

    # Combine subsets into a single dataframe
    df_out = pd.concat([df_train, df_test])
    df_out.sort_values(by=['Image path'], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    log.info('')
    log.info('Overall train/test split')
    log.info(
        f'Subjects..................: {df_train["Subject ID"].nunique()}/{df_test["Subject ID"].nunique()}',
    )
    log.info(
        f'Studies...................: {df_train["Study ID"].nunique()}/{df_test["Study ID"].nunique()}',
    )
    log.info(f'Objects...................: {len(df_train)}/{len(df_test)}')

    return df_out


def save_metadata(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    """Save metadata for an intermediate dataset.

    Args:
        df: dataframe with metadata for all images
        save_dir: directory where the output files will be saved
    Returns:
        None
    """
    df_path = os.path.join(save_dir, 'metadata.xlsx')
    logging.info(f'Saving metadata to {df_path}')
    df.sort_values(['Image path'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    df.to_excel(
        df_path,
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_sly_to_int',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    """Convert Supervisely dataset into Intermediate.

    Args:
        dataset_dir: a path to Supervisely dataset directory
        include_dirs: a list of subsets to include in the dataset
        exclude_dirs: a list of subsets to exclude from the dataset
        save_dir: directory where the output frontal images and metadata will be stored
    Returns:
        None
    """

    df = read_sly_project(
        dataset_dir=cfg.dataset_dir,
        include_dirs=cfg.include_dirs,
        exclude_dirs=cfg.exclude_dirs,
    )

    log.info('Processing annotations')
    groups = df.groupby(['img_path', 'ann_path'])
    processing_func = partial(
        process_sample,
        save_dir=cfg.save_dir,
    )
    result = Parallel(n_jobs=-1)(
        delayed(processing_func)(group)
        for group in tqdm(groups, desc='Dataset conversion', unit=' image')
    )
    df = pd.concat(result)

    df = split_dataset(
        df=df,
        train_size=cfg.train_size,
        seed=cfg.seed,
    )

    save_metadata(
        df=df,
        save_dir=cfg.save_dir,
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
