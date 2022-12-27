import argparse
import os

import fiftyone as fo

from settings import COCO_SAVE_DIR, SEED


def main(
    dataset_dir: str,
    subset: str,
    dataset_name: str = None,
    max_samples: int = None,
    shuffle: bool = False,
    seed: int = 11,
) -> None:

    """
    Args:
        dataset_dir: a directory with COCO dataset including train and test subsets
        subset: a name of subset i.e. train or test
        dataset_name: a name for the dataset
        max_samples: a maximum number of samples to import. By default, all samples are imported
        shuffle: whether to randomly shuffle the order in which the samples are imported
        seed: a random seed to use when shuffling

    Returns:
        None
    """

    subset_dir = os.path.join(dataset_dir, subset)
    try:
        dataset = fo.Dataset.from_dir(
            dataset_dir=subset_dir,
            dataset_type=fo.types.COCODetectionDataset,
            overwrite=True,
            persistent=False,
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed,
        )
    except ValueError:
        dataset = fo.load_dataset(dataset_name)
    session = fo.launch_app(dataset)
    session.wait()
    dataset.delete()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='COCO dataset visualization')
    parser.add_argument('--dataset_dir', default=COCO_SAVE_DIR, type=str)
    parser.add_argument('--subset', default='test', type=str)
    parser.add_argument('--dataset_name', default=None, type=str)
    parser.add_argument('--max_samples', default=None, type=int)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        subset=args.subset,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        seed=args.seed,
        shuffle=args.shuffle,
    )
