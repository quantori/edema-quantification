import os
import logging
from typing import List, Optional

import pandas as pd
import supervisely_lib as sly


def read_sly_project(
    dataset_dir: str,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """

    Args:
        dataset_dir: a path to Supervisely dataset directory
        include_dirs: a list of subsets to include in the dataset
        exclude_dirs: a list of subsets to exclude from the dataset

    Returns:
        df: dataframe representing the dataset
    """
    logging.info(f'Dataset dir..........: {dataset_dir}')
    assert os.path.exists(dataset_dir) and os.path.isdir(dataset_dir), 'Wrong project dir: {}'.format(dataset_dir)
    project = sly.Project(
        directory=dataset_dir,
        mode=sly.OpenMode.READ,
    )

    img_paths: List[str] = []
    ann_paths: List[str] = []
    subset_list: List[str] = []

    for dataset in project:
        subset = dataset.name

        if include_dirs and subset not in include_dirs:
            logging.info(f'Excluded dir.........: {subset}')
            continue

        if exclude_dirs and subset in exclude_dirs:
            logging.info(f'Excluded dir.........: {subset}')
            continue

        logging.info(f'Included dir.........: {subset}')
        for item_name in dataset:
            img_path, ann_path = dataset.get_item_paths(item_name)
            img_paths.append(img_path)
            ann_paths.append(ann_path)
            subset_list.append(subset)

    df = pd.DataFrame.from_dict(
        {
            'img_path': img_paths,
            'ann_path': ann_paths,
            'subset': subset_list,
        }
    )
    df.sort_values(['subset'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
