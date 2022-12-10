import json
import logging

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tools.utils_coco import *
from tools.utils import copy_files
from tools.utils_sly import FIGURE_MAP

from settings import INTERMEDIATE_DATASET_DIR, COCO_SAVE_DIR, TN_DIR, TRAIN_SIZE, BOX_EXTENSION

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
) -> pd.DataFrame:
    """

    Args:
        dataset_dir: path to directory containing series with images and labels inside

    Returns:
        metadata_short: data frame derived from a metadata file
    """

    metadata = pd.read_excel(os.path.join(dataset_dir, 'metadata.xlsx'))
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
    tn_dir: str,
    train_size: float,
    seed: int,
) -> dict:
    """

    Args:
        metadata_short: data frame derived from a metadata file
        tn_dir: directory placed outside the dataset_dir including images with no annotations
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

    train = metadata_short[metadata_short['Subject ID'].isin(train_ids['Subject ID'])]
    test = metadata_short[metadata_short['Subject ID'].isin(test_ids['Subject ID'])]

    subsets['test']['images'].extend(test['Image path'])
    subsets['test']['labels'].extend(test['Annotation path'])
    subsets['train']['images'].extend(train['Image path'])
    subsets['train']['labels'].extend(train['Annotation path'])

    if tn_dir is not None:
        tn_images, tn_labels = create_tn_subset(tn_dir)
        subsets['train']['images'].extend(tn_images)
        subsets['train']['labels'].extend(tn_labels)
    else:
        tn_images, tn_labels = [], []

    logger.info('')
    logger.info('Overall train/test split')
    logger.info(
        f'Subjects..................: {train["Subject ID"].nunique()}/{test["Subject ID"].nunique()}'
    )
    logger.info(
        f'Studies...................: {train["Study ID"].nunique()}/{test["Study ID"].nunique()}'
    )
    logger.info(f'Images....................: {len(train)}/{len(test)}')
    logger.info(f'Images in TN dataset......: {len(tn_images)}')

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
            ann_data, ann_id = get_ann_info_with_extension(
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
    tn_dir: str = None,
    train_size: float = 0.8,
    seed: int = 11,
) -> None:
    """

    Args:
        dataset_dir: path to directory containing series with images and labels inside
        save_dir: directory where split datasets are saved to
        tn_dir: directory placed outside the dataset_dir including images with no annotations
        train_size: a fraction used to split dataset into train and test subsets
        box_extension: a value used to extend or contract object box sizes
        seed: random value for splitting train and test subsets
    Returns:
        None
    """

    logger.info(f'Input directory...........: {dataset_dir}')
    logger.info(f'TN directory..............: {tn_dir}')
    logger.info(f'Train/Test split..........: {train_size:.2f} / {(1 - train_size):.2f}')
    logger.info(f'Box extension.............: {box_extension}')
    logger.info(f'Seed......................: {seed}')
    logger.info(f'Output directory..........: {save_dir}')

    if tn_dir is not None:
        assert Path(dataset_dir) not in Path(tn_dir).parents, 'tn_dir should be outside dataset_dir'

    metadata_short = get_metadata_info(dataset_dir)

    subsets = prepare_subsets(metadata_short, tn_dir, train_size, seed)

    prepare_coco(subsets, save_dir, box_extension)

    logger.info(f'Complete')


if __name__ == '__main__':
    main(
        dataset_dir=INTERMEDIATE_DATASET_DIR,
        save_dir=COCO_SAVE_DIR,
        box_extension=BOX_EXTENSION,
        tn_dir=TN_DIR,
        train_size=TRAIN_SIZE,
        seed=11,
    )