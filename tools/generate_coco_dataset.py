import json
import logging
import argparse
from pathlib import Path

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tools.utils_coco import *
from tools.utils import copy_files
from tools.utils_sly import FIGURE_MAP
from settings import (
    INTERMEDIATE_DATASET_DIR,
    EXCLUDE_CLASSES,
    TRAIN_SIZE,
    BOX_EXTENSION,
    SEED,
    COCO_SAVE_DIR,
)

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_metadata_info(
    dataset_dir: str,
    exclude_classes: List[str] = None,
) -> pd.DataFrame:
    """

    Args:
        dataset_dir: path to directory containing series with images and labels inside
        exclude_classes: a list of classes to exclude from the COCO dataset

    Returns:
        metadata_short: data frame derived from a metadata file
    """

    metadata = pd.read_excel(os.path.join(dataset_dir, 'metadata.xlsx'))
    metadata = metadata[~metadata['Class'].isin(exclude_classes)]
    metadata_short = metadata[
        [
            'Image path',
            'Annotation path',
            'Subject ID',
            'Study ID',
            'Figure ID',
            'Class ID',
        ]
    ].drop_duplicates()
    metadata_short = metadata_short.dropna(subset=['Class ID'])

    return metadata_short


def prepare_subsets(
    metadata_short: pd.DataFrame,
    train_size: float,
    seed: int,
) -> dict:
    """

    Args:
        metadata_short: data frame derived from a metadata file
        train_size: a fraction used to split dataset into train and test subsets
        seed: random value for splitting train and test subsets

    Returns:
        subsets: dictionary which contains image/annotation paths for train and test subsets
    """
    subsets = {
        'train': {'images': [], 'labels': []},
        'test': {'images': [], 'labels': []},
    }
    metadata_unique_subject_id = (
        metadata_short.groupby(by='Subject ID', as_index=False)['Class ID'].max().astype(int)
    )
    train_ids, test_ids = train_test_split(
        metadata_unique_subject_id,
        train_size=train_size,
        shuffle=True,
        random_state=seed,
        stratify=metadata_unique_subject_id['Class ID'],
    )

    df_train = metadata_short[metadata_short['Subject ID'].isin(train_ids['Subject ID'])]
    df_test = metadata_short[metadata_short['Subject ID'].isin(test_ids['Subject ID'])]

    mask_empty = df_test['Class ID'] == 0
    df_empty = df_test[mask_empty]
    df_test = df_test.drop(df_test.index[mask_empty])
    df_train = df_train.append(df_empty, ignore_index=True)
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    subsets['test']['images'].extend(df_test['Image path'])
    subsets['test']['labels'].extend(df_test['Annotation path'])
    subsets['train']['images'].extend(df_train['Image path'])
    subsets['train']['labels'].extend(df_train['Annotation path'])

    logger.info('')
    logger.info('Overall train/test split')
    logger.info(
        f'Subjects..................: {df_train["Subject ID"].nunique()}/{df_test["Subject ID"].nunique()}'
    )
    logger.info(
        f'Studies...................: {df_train["Study ID"].nunique()}/{df_test["Study ID"].nunique()}'
    )
    logger.info(f'Images....................: {len(df_train)}/{len(df_test)}')

    assert len(subsets['train']['images']) == len(
        subsets['train']['labels']
    ), 'Mismatch length of the training subset'
    assert len(subsets['test']['images']) == len(
        subsets['test']['labels']
    ), 'Mismatch length of the testing subset'

    return subsets


def prepare_coco(
    subsets: dict,
    save_dir: str,
    box_extension: dict,
) -> None:
    """

    Args:
        subsets: dictionary which contains image/annotation paths for train and test subsets
        save_dir: directory where split datasets are saved to
        box_extension: a value used to extend or contract object box sizes

    Returns:
        None
    """
    categories_coco = []
    for idx, (key, value) in enumerate(FIGURE_MAP.items()):
        categories_coco.append({'id': value, 'name': key})

    for subset_name, subset in subsets.items():
        imgs_coco = []
        anns_coco = []
        ann_id = 0
        for img_id, (img_path, ann_path) in tqdm(
            enumerate(zip(subset['images'], subset['labels'])),
            desc=f'{subset_name.capitalize()} subset processing',
            unit=' sample',
        ):
            img_data = get_img_info(
                img_path=img_path,
                img_id=img_id,
            )
            # TODO: fix the function in a way that processes images with no labels i.e. healthy patients
            ann_data, ann_id = get_ann_info(
                label_path=ann_path,
                img_id=img_id,
                ann_id=ann_id,
                box_extension=box_extension,
            )
            imgs_coco.append(img_data)
            anns_coco.extend(ann_data)

        dataset = {
            'images': imgs_coco,
            'annotations': anns_coco,
            'categories': categories_coco,
        }

        save_img_dir = os.path.join(save_dir, subset_name, 'data')
        copy_files(file_list=subset['images'], save_dir=save_img_dir)
        save_ann_path = os.path.join(save_dir, subset_name, 'labels.json')
        with open(save_ann_path, 'w') as file:
            json.dump(dataset, file)


def main(
    dataset_dir: str,
    save_dir: str,
    box_extension: dict,
    exclude_classes: List[str] = None,
    train_size: float = 0.8,
    seed: int = 11,
) -> None:
    """

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

    logger.info(f'Input directory...........: {dataset_dir}')
    logger.info(f'Excluded classes...........: {exclude_classes}')
    logger.info(f'Train/Test split..........: {train_size:.2f} / {(1 - train_size):.2f}')
    logger.info(f'Box extension.............: {box_extension}')
    logger.info(f'Seed......................: {seed}')
    logger.info(f'Output directory..........: {save_dir}')

    metadata_short = get_metadata_info(dataset_dir, exclude_classes)

    subsets = prepare_subsets(metadata_short, train_size, seed)

    prepare_coco(subsets, save_dir, box_extension)

    logger.info(f'Complete')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Intermediate-to-COCO dataset conversion')
    parser.add_argument('--dataset_dir', default=INTERMEDIATE_DATASET_DIR, type=str)
    parser.add_argument('--exclude_classes', default=EXCLUDE_CLASSES, type=str)
    parser.add_argument('--train_size', default=TRAIN_SIZE, type=float)
    parser.add_argument('--box_extension', default=BOX_EXTENSION)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--save_dir', default=COCO_SAVE_DIR, type=str)
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        exclude_classes=args.exclude_classes,
        train_size=args.train_size,
        box_extension=args.box_extension,
        seed=args.seed,
        save_dir=args.save_dir,
    )
