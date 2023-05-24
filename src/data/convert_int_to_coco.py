import json
import logging
import os
from typing import List

import albumentations as A
import cv2
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.utils import copy_files
from src.data.utils_coco import get_ann_info, get_img_info
from src.data.utils_sly import FEATURE_MAP, FEATURE_MAP_REVERSED, get_box_sizes

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _modify_box_geometry(
    df: pd.DataFrame,
    box_extension: dict,
) -> pd.DataFrame:
    for idx in tqdm(df.index, desc='Modify box geometry', unit=' boxes'):
        box_extension_feature = box_extension[FEATURE_MAP_REVERSED[df.at[idx, 'Feature ID']]]
        df.at[idx, 'x1'] -= box_extension_feature[0]
        df.at[idx, 'y1'] -= box_extension_feature[1]
        df.at[idx, 'x2'] += box_extension_feature[0]
        df.at[idx, 'y2'] += box_extension_feature[1]
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


def _modify_image_geometry(
    df: pd.DataFrame,
    output_size: List[int],
) -> pd.DataFrame:
    # TODO: update sizes of images and boxes
    transform = A.Compose(
        [
            A.LongestMaxSize(
                max_size=max(output_size),
                interpolation=1,
                p=1.0,
            ),
            A.PadIfNeeded(
                min_width=output_size[0],
                min_height=output_size[1],
                position='center',
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            ),
            # FIXME: CenterCrop sometimes cut out a part of the lungs
            A.CenterCrop(
                width=output_size[0],
                height=output_size[1],
            ),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['class_labels'],
        ),
    )

    for idx in tqdm(df.index):
        img_path = df.at[idx, 'Image path']
        img = cv2.imread(img_path)
        box = [
            int(df.at[idx, 'x1']),
            int(df.at[idx, 'y1']),
            int(df.at[idx, 'x2']),
            int(df.at[idx, 'y2']),
        ]
        feature = df.at[idx, 'Feature']

        trans = transform(
            image=img,
            bboxes=[box],
            class_labels=[feature],
        )
        img_trans = trans['image']
        box_trans = [round(val) for val in trans['bboxes'][0]]
        print(img_trans)
        print(box_trans)

    return df


def process_metadata(
    dataset_dir: str,
    output_size: List[int],
    box_extension: dict,
    excluded_features: List[str] = None,
) -> pd.DataFrame:
    """Process dataset metadata.

    Args:
        dataset_dir: a path to the directory containing series with images and labels
        output_size: a list specifying the desired image size
        box_extension: a dictionary specifying box offsets for each feature
        excluded_features: a list of features to be excluded from the COCO dataset
    Returns:
        metadata: an updated metadata dataframe
    """
    metadata = pd.read_excel(os.path.join(dataset_dir, 'metadata.xlsx'))
    metadata = metadata[metadata['View'] == 'Frontal']
    metadata = metadata[~metadata['Feature'].isin(excluded_features)]
    metadata = metadata.dropna(subset=['Class ID'])
    metadata = _modify_box_geometry(
        df=metadata,
        box_extension=box_extension,
    )
    if output_size:
        assert len(output_size) == 2 and all(
            isinstance(val, int) for val in output_size
        ), 'output_size must be a list of two integers'
        metadata = _modify_image_geometry(
            df=metadata,
            output_size=output_size,
        )

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


def prepare_coco(
    df: pd.DataFrame,
    box_extension: dict,
    save_dir: str,
) -> pd.DataFrame:
    """Prepare and save training and test subsets in COCO format.

    Args:
        df: dataframe containing information about the training and test subsets
        box_extension: a value used to extend or contract object box sizes
        save_dir: directory where split datasets are stored
    Returns:
        df: updated COCO dataframe with training and test subsets
    """
    categories_coco = []
    for idx, (key, value) in enumerate(FEATURE_MAP.items()):
        categories_coco.append({'id': value, 'name': key})

    # Iterate over subsets
    subset_list = list(df['Split'].unique())
    for subset in subset_list:
        df_subset = df[df['Split'] == subset]
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

    # Update dataframe structure
    columns_to_drop = [
        'Source type',
        'Reference type',
        'Match',
        'Mask',
        'Points',
    ]
    df = df.drop(labels=columns_to_drop, axis=1)
    df['Image path'] = df.apply(
        lambda row: os.path.join(save_dir, row['Split'], 'data', row['Image name']),
        axis=1,
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

    return df


def save_subset_metadata(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    df['Confidence'] = 1.0
    subset_list = list(df['Split'].unique())
    for subset in subset_list:
        df_subset = df[df['Split'] == subset]
        df_subset = df_subset.drop(labels='Split', axis=1)
        df_subset.reset_index(drop=True, inplace=True)
        save_path = os.path.join(save_dir, f'{subset}', 'labels.xlsx')
        df_subset.index += 1
        df_subset.to_excel(
            save_path,
            sheet_name='Metadata',
            index=True,
            index_label='ID',
        )
        log.info(f'{subset.capitalize()} labels saved to.....: {save_path}')


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
        excluded_features: a list of features to exclude from the COCO dataset
        train_size: a fraction used to split dataset into train and test subsets
        box_extension: a value used to extend or contract object box sizes
        seed: random value for splitting train and test subsets
    Returns:
        None
    """
    log.info(f'Input directory...........: {cfg.dataset_dir}')
    log.info(f'Output size...............: {cfg.output_size}')
    log.info(f'Excluded features.........: {cfg.excluded_features}')
    log.info(f'Train/Test split..........: {cfg.train_size:.2f} / {(1 - cfg.train_size):.2f}')
    log.info(f'Box extension.............: {cfg.box_extension}')
    log.info(f'Seed......................: {cfg.seed}')
    log.info(f'Output directory..........: {cfg.save_dir}')

    metadata = process_metadata(
        dataset_dir=cfg.dataset_dir,
        output_size=cfg.output_size,
        box_extension=cfg.box_extension,
        excluded_features=cfg.excluded_features,
    )

    metadata_split = split_dataset(
        metadata=metadata,
        train_size=cfg.train_size,
        seed=cfg.seed,
    )

    # TODO: remove box_extension, as it is used in process_metadata
    df = prepare_coco(
        df=metadata_split,
        box_extension=cfg.box_extension,
        save_dir=cfg.save_dir,
    )

    save_subset_metadata(
        df=df,
        save_dir=cfg.save_dir,
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
