import json
import logging
import argparse

from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tools.utils_coco import *
from tools.utils import get_file_list, copy_files

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(
    dataset_dir: str,
    save_dir: str,
    tn_dir: str = None,
    train_size: float = 0.8,
    box_extension: int = 0,
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
    logger.info(f'Train/Test split..........: {train_size}/{1 - train_size}')
    logger.info(f'Box extension.............: {box_extension}')
    logger.info(f'Seed......................: {seed}')
    logger.info(f'Output directory..........: {save_dir}')

    if tn_dir is not None:
        assert Path(dataset_dir) not in Path(tn_dir).parents, 'tn_dir should not be inside the data_dirs'

    subsets = {
        'train': {'images': [], 'labels': []},
        'test': {'images': [], 'labels': []},
    }

    # TODO: split subsets based on patient level
    _series_paths = glob(dataset_dir + '/*/')
    series_paths.extend(_series_paths)
    train_studies, test_studies = train_test_split(
        series_paths, test_size=train_size, shuffle=True, random_state=seed
    )

    # TODO: add images and annotiation paths to subsets
    # Processing of the training and validation subsets
    train_val_images = get_file_list(
        src_dirs=train_studies,
        ext_list=['.png', '.jpg', '.jpeg', '.bmp']
    )
    train_val_labels = get_file_list(
        src_dirs=train_studies,
        ext_list=['.txt']
    )
    train_images, val_images = train_test_split(
        train_val_images,
        test_size=train_size,
        shuffle=True,
        random_state=seed
    )
    train_labels, val_labels = train_test_split(
        train_val_labels,
        train_size=train_size,
        shuffle=True,
        random_state=seed
    )

    subsets['val']['images'].extend(val_images)         # TODO: keep only train and test
    subsets['val']['labels'].extend(val_labels)         # TODO: keep only train and test
    subsets['train']['images'].extend(train_images)
    subsets['train']['labels'].extend(train_labels)

    # Processing of the testing subset
    test_images = get_file_list(
        src_dirs=test_studies,
        ext_list=['.png', '.jpg', '.jpeg', '.bmp']
    )
    test_labels = get_file_list(
        src_dirs=test_studies,
        ext_list=['.txt']
    )
    subsets['test']['images'].extend(test_images)
    subsets['test']['labels'].extend(test_labels)

    logger.info(
        'Training dataset    : {:d} images'.format(len(train_images))
    )
    logger.info(
        'Validation dataset  : {:d} images'.format(len(val_images))
    )
    logger.info(
        'Testing dataset     : {:d} images\n'.format(len(test_images))
    )

    # TODO: add True Negative dataset
    if tn_dir is not None:
        tn_images, tn_labels = create_tn_subset(tn_dir)
        subsets['train']['images'].extend(tn_images)
        subsets['train']['labels'].extend(tn_labels)
    else:
        tn_images, tn_labels = [], []

    logger.info('Overall dataset')
    logger.info(
        'Training dataset    : {:d} series, {:d} images, {:d} labels '.format(
            len(train_studies), len(subsets['train']['images']), len(subsets['train']['labels'])
        )
    )
    logger.info(
        'TNs in training     : {:d} series, {:d} images, {:d} labels '.format(
            0 if tn_dir is None else 1, len(tn_images), len(tn_labels)
        )
    )
    logger.info(
        'Validation dataset  : {:d} series, {:d} images, {:d} labels '.format(
            len(train_studies), len(subsets['val']['images']), len(subsets['val']['labels'])
        )
    )
    logger.info(
        'Testing dataset     : {:d} series, {:d} images, {:d} labels '.format(
            len(test_studies), len(subsets['test']['images']), len(subsets['test']['labels'])
        )
    )

    # TODO: use all categories
    categories = [
        {
            'id': 0,
            'name': 'Cephalization',
        },
        {
            'id': 1,
            'name': 'Artery'},
    ]

    assert len(subsets['train']['images']) == len(
        subsets['train']['labels']
    ), 'Mismatch length of the training subset'
    assert len(subsets['val']['images']) == len(
        subsets['val']['labels']
    ), 'Mismatch length of the validation subset'
    assert len(subsets['test']['images']) == len(
        subsets['test']['labels']
    ), 'Mismatch length of the testing subset'

    # TODO: aggregate all images and annotations, and save them
    for subset_name, subset in subsets.items():
        imgs_coco = []
        anns_coco = []
        ann_id = 0
        for img_id, (img_path, ann_path) in tqdm(
                enumerate(zip(subset['images'], subset['labels'])),
                desc=f'{subset_name.capitalize()} subset processing',
                unit=' samples'
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
            'categories': categories
        }

        save_img_dir = os.path.join(save_dir, subset_name, 'data')
        copy_files(file_list=subset['images'], save_dir=save_img_dir)
        save_ann_path = os.path.join(save_dir, subset_name, 'labels.json')
        with open(save_ann_path, 'w') as file:
            json.dump(dataset, file)


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
