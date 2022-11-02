import os
import logging
from typing import List, Optional, Union

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


def get_figure_info(
        name_or_id: Union[int, str]
) -> Union[int, str]:
    """
    Args:
        name_or_id: figure name or ID

    Returns:
        value: figure ID or name
    """

    figure_map = {
        'Cephalization': 0,
        'Heart': 1,
        'Artery': 2,
        'Bronchus': 3,
        'Kerley': 4,
        'Cuffing': 5,
        'Effusion': 6,
        'Bat': 7,
        'Infiltrate': 8,
    }

    if isinstance(name_or_id, int):
        for key, val in figure_map.items():
            if val == name_or_id:
                return key
        raise ValueError(f'No key with ID {name_or_id}')
    elif isinstance(name_or_id, str):
        name_or_id = name_or_id.capitalize()
        try:
            return figure_map[name_or_id]
        except:
            raise KeyError(f'Invalid field value: {name_or_id}')
    else:
        raise TypeError(f'Invalid field value: {type(name_or_id)}')
