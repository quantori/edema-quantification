import json
from typing import List, Tuple, Dict, Union

import wandb
import numpy as np
from sklearn.model_selection import train_test_split


def normalize_image(image: np.ndarray,
                    target_min: Union[int, float] = 0.0,
                    target_max: Union[int, float] = 1.0,
                    target_type=np.float32) -> Union[int, float]:
    a = (target_max - target_min) / (image.max() - image.min())
    b = target_max - a * image.max()
    image_norm = (a * image + b).astype(target_type)
    return image_norm


def split_data(img_paths: List[str],
               ann_paths: List[str],
               dataset_names: List[str],
               class_name: str,
               seed: int = 11,
               ratio: List[float] = (0.8, 0.1, 0.1),
               normal_datasets: List[str] = ('rsna_normal', 'chest_xray_normal')) -> Dict:

    assert sum(ratio) <= 1, 'The sum of ratio values should not be greater than 1'
    output = {'train': Tuple[List[str], List[str]],
              'val': Tuple[List[str], List[str]],
              'test': Tuple[List[str], List[str]]}
    img_paths_train: List[str] = []
    ann_paths_train: List[str] = []
    img_paths_val: List[str] = []
    ann_paths_val: List[str] = []
    img_paths_test: List[str] = []
    ann_paths_test: List[str] = []

    train_ratio = ratio[0]
    for dataset_name in dataset_names:
        img_paths_ds = list(filter(lambda path: dataset_name in path, img_paths))
        ann_paths_ds = list(filter(lambda path: dataset_name in path, ann_paths))

        if dataset_name not in normal_datasets:
            img_paths_ds, ann_paths_ds = drop_empty_annotations(img_paths=img_paths_ds,
                                                                ann_paths=ann_paths_ds,
                                                                class_name=class_name)

        # validation fraction = 0
        if ratio[1] == 0 and ratio[2] > 0:
            x_train, x, y_train, y = train_test_split(img_paths_ds, ann_paths_ds, train_size=train_ratio, random_state=seed)
            x_val, y_val = [], []
            x_test, y_test = x, y
        # test fraction = 0
        elif ratio[1] > 0 and ratio[2] == 0:
            x_train, x, y_train, y = train_test_split(img_paths_ds, ann_paths_ds, train_size=train_ratio, random_state=seed)
            x_val, y_val = x, y
            x_test, y_test = [], []
        # validation and test fractions > 0
        elif ratio[1] > 0 and ratio[2] > 0:
            x_train, x, y_train, y = train_test_split(img_paths_ds, ann_paths_ds, train_size=train_ratio, random_state=seed)
            test_ratio = ratio[2] / (ratio[1] + ratio[2])
            x_val, x_test, y_val, y_test = train_test_split(x, y, test_size=test_ratio, random_state=seed)
        # validation and test fractions = 0
        elif ratio[1] == 0 and ratio[2] == 0:
            x_train, y_train = img_paths_ds, ann_paths_ds
            x_val, y_val = [], []
            x_test, y_test = [], []
        else:
            raise ValueError('Incorrect ratio values!')

        img_paths_train.extend(x_train)
        ann_paths_train.extend(y_train)
        img_paths_val.extend(x_val)
        ann_paths_val.extend(y_val)
        img_paths_test.extend(x_test)
        ann_paths_test.extend(y_test)

    output['train'] = img_paths_train, ann_paths_train
    output['val'] = img_paths_val, ann_paths_val
    output['test'] = img_paths_test, ann_paths_test

    return output


def drop_empty_annotations(img_paths: List[str],
                           ann_paths: List[str],
                           class_name: str) -> Tuple[List[str], List[str]]:
    img_paths_cleaned: List[str] = []
    ann_paths_cleaned: List[str] = []
    for img_path, ann_path in zip(img_paths, ann_paths):
        with open(ann_path) as json_file:
            data = json.load(json_file)
        for obj in data['objects']:
            if obj['classTitle'] == class_name:
                img_paths_cleaned.append(img_path)
                ann_paths_cleaned.append(ann_path)
                break
    return img_paths_cleaned, ann_paths_cleaned


def get_logging_labels(class_names: List[str]) -> Dict[int, str]:
    l = {0: 'Background'}
    if 'Background' in class_names:
        class_names.remove('Background')
    for i, label in enumerate(class_names, 1):
        l[i] = label
    return l


def log_dataset(run, datasets_list, artefact_name):
    artifact = wandb.Artifact(artefact_name, type='dataset')
    for dataloader in datasets_list:
        img_paths, ann_paths = dataloader.dataset.img_paths, dataloader.dataset.ann_paths
        for img, ann in zip(img_paths, ann_paths):
            artifact.add_file(img)
            artifact.add_file(ann)
    run.log_artifact(artifact)


def convert_seconds_to_hms(sec: Union[float, int]) -> str:
    """Function that converts time period in seconds into %h:%m:%s expression.
    Args:
        sec (float): time period in seconds
    Returns:
        output (string): formatted time period
    """
    sec = int(sec)
    h = sec // 3600
    m = sec % 3600 // 60
    s = sec % 3600 % 60
    output = '{:02d}h:{:02d}m:{:02d}s'.format(h, m, s)
    return output


if __name__ == '__main__':

    # The code snippet below is used only for debugging
    labels = get_logging_labels(['COVID-19'])
    print(labels)
