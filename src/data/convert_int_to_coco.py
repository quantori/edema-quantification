import json
import logging
import os
from typing import List

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.utils import copy_files
from src.data.utils_coco import get_ann_info, get_img_info
from src.data.utils_sly import FIGURE_MAP

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def process_metadata(
    dataset_dir: str,
    exclude_classes: List[str] = None,
) -> pd.DataFrame:
    """Extract additional meta.

    Args:
        dataset_dir: path to directory containing series with images and labels inside
        exclude_classes: a list of classes to exclude from the COCO dataset
    Returns:
        meta: data frame derived from a meta file
    """
    metadata = pd.read_excel(os.path.join(dataset_dir, 'metadata.xlsx'))
    metadata = metadata[~metadata['Class'].isin(exclude_classes)]
    metadata = metadata.dropna(subset=['Class ID'])

    return metadata


def split_dataset(
    metadata: pd.DataFrame,
    train_size: float,
    seed: int,
) -> pd.DataFrame:
    """Split dataset with stratification into training and test subsets.

    Args:
        metadata: data frame derived from a source metadata file
        train_size: a fraction used to split dataset into train and test subsets
        seed: random value for splitting train and test subsets
    Returns:
        subsets: dictionary which contains image/annotation paths for train and test subsets
    """
    # Extract a subset of unique subject IDs with stratification
    metadata_unique_subjects = (
        metadata.groupby(by='Subject ID', as_index=False)['Class ID'].max().astype(int)
    )

    # Split dataset into train and test subsets with
    train_ids, test_ids = train_test_split(
        metadata_unique_subjects,
        train_size=train_size,
        shuffle=True,
        random_state=seed,
        stratify=metadata_unique_subjects['Class ID'],
    )

    # Extract train and test subsets by indices
    df_train = metadata[metadata['Subject ID'].isin(train_ids['Subject ID'])]
    df_test = metadata[metadata['Subject ID'].isin(test_ids['Subject ID'])]

    # Move cases without edema from test to train
    mask_empty = df_test['Class ID'] == 0
    df_empty = df_test[mask_empty]
    df_test = df_test.drop(df_test.index[mask_empty])
    df_train = df_train.append(df_empty, ignore_index=True)
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    # Add split column
    df_train = df_train.assign(Split='train')
    df_test = df_test.assign(Split='test')

    # Combine subsets into a single dataframe
    df_out = pd.concat([df_train, df_test])
    df_out.drop('ID', axis=1, inplace=True)
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
    log.info(f'Images....................: {len(df_train)}/{len(df_test)}')

    return df_out


def prepare_subset(
    save_dir: str,
    subset: str,
    df_subset: pd.DataFrame,
    box_extension: dict,
    categories_coco: list,
) -> None:
    imgs_coco = []
    anns_coco = []
    ann_id = 0
    save_img_dir = os.path.join(save_dir, subset, 'data')
    os.makedirs(save_img_dir, exist_ok=True)
    img_groups = df_subset.groupby('Image path')

    for img_id, (img_path, sample) in tqdm(
        enumerate(img_groups),
        desc=f'{subset.capitalize()} subset processing',
        unit=' images',
    ):
        img_data = get_img_info(
            img_path=img_path,
            img_id=img_id,
        )
        imgs_coco.append(img_data)

        ann_data, ann_id = get_ann_info(
            df=sample,
            img_id=img_id,
            ann_id=ann_id,
            box_extension=box_extension,
        )

        if len(ann_data) > 0:
            anns_coco.extend(ann_data)

    dataset = {
        'images': imgs_coco,
        'annotations': anns_coco,
        'categories': categories_coco,
    }

    # Save JSONs with annotations
    save_img_dir = os.path.join(save_dir, subset, 'data')
    copy_files(file_list=list(df_subset['Image path']), save_dir=save_img_dir)
    save_ann_path = os.path.join(save_dir, subset, 'labels.json')
    with open(save_ann_path, 'w') as file:
        json.dump(dataset, file)


def get_categories_coco() -> list:
    categories_coco = []
    for idx, (key, value) in enumerate(FIGURE_MAP.items()):
        categories_coco.append({'id': value, 'name': key})
    return categories_coco


def prepare_coco(
    df: pd.DataFrame,
    box_extension: dict,
    save_dir: str,
) -> None:
    """Prepare and save training and test subsets in COCO format.

    Args:
        df: dataframe containing information about the training and test subsets
        box_extension: a value used to extend or contract object box sizes
        save_dir: directory where split datasets are stored
    Returns:
        None
    """
    categories_coco = get_categories_coco()

    # Iterate over subsets
    subset_list = list(df['Split'].unique())
    for subset in subset_list:
        df_subset = df[df['Split'] == subset]
        prepare_subset(
            save_dir=save_dir,
            subset=subset,
            df_subset=df_subset,
            box_extension=box_extension,
            categories_coco=categories_coco,
        )

    # Save COCO metadata
    save_path = os.path.join(save_dir, 'metadata.xlsx')
    df.index += 1
    df.to_excel(
        save_path,
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_int_to_coco',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    """Convert intermediate dataset to COCO.

    Args:
        dataset_dir: path to directory containing series with images and labels inside
        save_dir: directory where split datasets are saved to
        exclude_classes: a list of classes to exclude from the COCO dataset
        train_size: a fraction used to split dataset into train and test subsets
        box_extension: a value used to extend or contract object box sizes
        seed: random value for splitting train and test subsets
    Returns:
        None
    """
    log.info(f'Input directory...........: {cfg.dataset_dir}')
    log.info(f'Excluded classes...........: {cfg.exclude_classes}')
    log.info(f'Train/Test split..........: {cfg.train_size:.2f} / {(1 - cfg.train_size):.2f}')
    log.info(f'Box extension.............: {cfg.box_extension}')
    log.info(f'Seed......................: {cfg.seed}')
    log.info(f'Output directory..........: {cfg.save_dir}')

    metadata = process_metadata(cfg.dataset_dir, cfg.exclude_classes)

    metadata_split = split_dataset(metadata, cfg.train_size, cfg.seed)

    prepare_coco(metadata_split, cfg.box_extension, cfg.save_dir)

    log.info('Complete')


if __name__ == '__main__':
    main()
