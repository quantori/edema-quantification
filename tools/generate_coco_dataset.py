import json
import logging
import argparse

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tools.utils_coco import *
from tools.utils import copy_files
from tools.utils_sly import FIGURE_MAP

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
        metadata_short: a data frame with updated
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
    metadata: pd.DataFrame,
    tn_dir: str,
    train_size: float,
    seed: int,
) -> dict:

    subsets = {
        'train': {'images': [], 'labels': []},
        'test': {'images': [], 'labels': []},
    }

    # TODO: Incorrect split
    # TODO: (a) train and test subsets should not have samples of the same Subject ID
    # TODO: (b) y_train and y_test include IDs of figures rather than IDs of edema class
    X_train, X_test, y_train, y_test = train_test_split(
        metadata[['Image path', 'Annotation path', 'Subject ID', 'Study ID', 'Class ID']],
        metadata['Figure ID'],
        train_size=train_size,
        shuffle=True,
        random_state=seed,
        # TODO: to think if stratify works well during such a split
        # TODO: not sure if we can stratify by Class ID considering unique Subject IDs between train and test subsets
        stratify=metadata['Class ID'],
    )

    subsets['test']['images'].extend(X_test['Image path'])
    subsets['test']['labels'].extend(X_test['Annotation path'])
    subsets['train']['images'].extend(X_train['Image path'])
    subsets['train']['labels'].extend(X_train['Annotation path'])

    # TODO: In our case, a True Negative dataset is a dataset which
    # TODO: (a) doesn't have label files
    # TODO: (b) includes images where edema features are not presented
    # TODO: (c) is used for training only (not testing)
    if tn_dir is not None:
        tn_images, tn_labels = create_tn_subset(tn_dir)
        subsets['train']['images'].extend(tn_images)
        subsets['train']['labels'].extend(tn_labels)
    else:
        tn_images, tn_labels = [], []

    logger.info('')
    logger.info('Overall train/test split')
    logger.info(f'Subjects..................: {X_train["Subject ID"].nunique()}/{X_test["Subject ID"].nunique()}')
    logger.info(f'Studies...................: {X_train["Study ID"].nunique()}/{X_test["Study ID"].nunique()}')
    logger.info(f'Images....................: {len(X_train)}/{len(X_test)}')
    logger.info(f'Images in TN dataset......: {len(tn_images)}')

    assert len(subsets['train']['images']) == len(subsets['train']['labels']), 'Mismatch length of the training subset'
    assert len(subsets['test']['images']) == len(subsets['test']['labels']), 'Mismatch length of the testing subset'

    return subsets


def prepare_coco(
    subsets: dict,
    save_dir: str,
) -> None:

    # TODO: (a) copy all images to a specific dir i.e. 'data' in our case
    # TODO: (b) create two json files for train and test subsets
    # FIXME: Kindly ask you to use one dictionary with classes and figures rather than creating different ones
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
            ann_data, ann_id = get_ann_info(
                label_path=ann_path,
                img_id=img_id,
                ann_id=ann_id,
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
    tn_dir: str = None,
    train_size: float = 0.8,
    box_extension: int = 0,         # TODO: implement box extension (not this feature is not working)
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
        assert (
            Path(dataset_dir) not in Path(tn_dir).parents
        ), 'tn_dir should be outside dataset_dir'

    metadata_short = get_metadata_info(dataset_dir)

    subsets = prepare_subsets(metadata_short, tn_dir, train_size, seed)

    prepare_coco(subsets, save_dir)

    logger.info(f'Complete')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare split subsets')
    parser.add_argument('--dataset_dir', default='dataset/MIMIC-CXR-Edema-Intermediate', type=str)
    parser.add_argument('--tn_dir', default=None, type=str)
    parser.add_argument('--train_size', default=0.8, type=float)
    parser.add_argument('--box_extension', default=0, type=int)
    parser.add_argument('--save_dir', default='dataset/MIMIC-CXR-Edema-COCO', type=str)
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        tn_dir=args.tn_dir,
        train_size=args.train_size,
        box_extension=args.box_extension,
        save_dir=args.save_dir,
    )
