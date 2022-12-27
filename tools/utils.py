import os
import logging
import warnings
import multiprocessing
from pathlib import Path
from functools import partial
from PIL import Image, ImageFilter
from typing import Dict, List, Union, Tuple

import cv2
import shutil
import numpy as np
from tqdm import tqdm


def get_file_list(
    src_dirs: Union[List[str], str],
    ext_list: Union[List[str], str],
    include_template: str = '',
) -> List[str]:
    """
    Get list of files with the specified extensions

    Args:
        src_dirs: directory(s) with files inside
        ext_list: extension(s) used for a search
        include_template: include directories with this template
    Returns:
        all_files: a list of file paths
    """

    all_files = []
    src_dirs = [src_dirs, ] if isinstance(src_dirs, str) else src_dirs
    ext_list = [ext_list, ] if isinstance(ext_list, str) else ext_list
    for src_dir in src_dirs:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_ext = Path(file).suffix
                file_ext = file_ext.lower()
                dir_name = os.path.basename(root)
                if (
                    file_ext in ext_list
                    and include_template in dir_name
                ):
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
    all_files.sort()
    return all_files


def copy_single_file(
    file_path: str,
    save_dir: str
) -> None:
    try:
        shutil.copy(file_path, save_dir)
    except Exception as e:
        logging.info(f'Exception: {e}\nCould not copy {file_path}')


def copy_files(
    file_list: List[str],
    save_dir: str
) -> None:
    os.makedirs(save_dir) if not os.path.isdir(save_dir) else False
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    copy_func = partial(
        copy_single_file,
        save_dir=save_dir
    )
    pool.map(
        copy_func,
        tqdm(file_list, desc='Copy files', unit=' files'))
    pool.close()


def convert_seconds_to_hms(
    sec: Union[float, int],
) -> str:
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


def separate_lungs(
    mask: np.array,
):
    assert (
        np.max(mask) <= 1
        and np.min(mask) >= 0
    ), f'Mask values should be in [0,1] scale (max: {np.max(mask)}, min: {np.min(mask)}'
    binary_map = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image=binary_map,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    centroids = centroids.astype(np.int32)
    lungs = []

    if num_labels != 3:
        warnings.warn('There are no 2 objects on the predicted mask, this might cause incorrect results')

        while num_labels <= 2:
            stats = np.append(stats, [stats[-1]], axis=0)
            centroids = np.append(centroids, [centroids[-1]], axis=0)
            num_labels += 1

    for i in range(1, 3):
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        x1, y1 = x0 + stats[i, cv2.CC_STAT_WIDTH], y0 + stats[i, cv2.CC_STAT_HEIGHT]
        zero_matrix = np.zeros_like(mask)
        zero_matrix[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        lungs.append({"lung": zero_matrix, "centroid": centroids[i]})

    if lungs[0]["centroid"][0] < lungs[1]["centroid"][0]:
        left_lung, right_lung = lungs[0]["lung"], lungs[1]["lung"]
    else:
        right_lung, left_lung = lungs[0]["lung"], lungs[1]["lung"]
    return left_lung, right_lung


def extract_model_params(
    model_path: str,
) -> Dict:

    models = [
        "Unet",
        "Unet++",
        "DeepLabV3",
        "DeepLabV3+",
        "FPN",
        "Linknet",
        "PSPNet",
        "PAN",
        "MAnet",
    ]

    encoders = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x4d",
        "resnext101_32x8d",
        "resnext101_32x16d",
        "resnext101_32x32d",
        "resnext101_32x48d",
        "timm-resnest14d",
        "timm-resnest26d",
        "timm-resnest50d",
        "timm-resnest101e",
        "timm-resnest200e",
        "timm-resnest269e",
        "timm-resnest50d_4s2x40d",
        "timm-resnest50d_1s4x24d",
        "timm-res2net50_26w_4s",
        "timm-res2net101_26w_4s",
        "timm-res2net50_26w_8s",
        "timm-res2net50_48w_2s",
        "timm-res2net50_14w_8s",
        "timm-res2next50",
        "timm-regnetx_016",
        "timm-regnetx_032",
        "timm-res2net50_26w_6s",
        "timm-regnetx_002",
        "timm-regnetx_004",
        "timm-regnetx_006",
        "timm-regnetx_008",
        "timm-regnetx_040",
        "timm-regnetx_064",
        "timm-regnetx_080",
        "timm-regnetx_120",
        "timm-regnetx_160",
        "timm-regnetx_320",
        "timm-regnety_002",
        "timm-regnety_004",
        "timm-regnety_006",
        "timm-regnety_008",
        "timm-regnety_016",
        "timm-regnety_032",
        "timm-regnety_040",
        "timm-regnety_064",
        "timm-regnety_080",
        "timm-regnety_120",
        "timm-regnety_160",
        "timm-regnety_320",
        "senet154",
        "se_resnet50",
        "se_resnet101",
        "se_resnet152",
        "se_resnext50_32x4d",
        "se_resnext101_32x4d",
        "timm-skresnet18",
        "timm-skresnet34",
        "timm-skresnext50_32x4d",
        "densenet121",
        "densenet169",
        "densenet201",
        "densenet161",
        "inceptionresnetv2",
        "inceptionv4",
        "xception",
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
        "efficientnet-b6",
        "efficientnet-b7",
        "timm-efficientnet-b0",
        "timm-efficientnet-b1",
        "timm-efficientnet-b2",
        "timm-efficientnet-b3",
        "timm-efficientnet-b4",
        "timm-efficientnet-b5",
        "timm-efficientnet-b6",
        "timm-efficientnet-b7",
        "timm-efficientnet-b8",
        "timm-efficientnet-l2",
        "timm-efficientnet-lite0",
        "timm-efficientnet-lite1",
        "timm-efficientnet-lite2",
        "timm-efficientnet-lite3",
        "timm-efficientnet-lite4",
        "mobilenet_v2",
        "dpn68",
        "dpn68b",
        "dpn92",
        "dpn98",
        "dpn107",
        "dpn131",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
    ]

    weights = [
        "imagenet",
        "ssl",
        "swsl",
        "instagram",
        "imagenet+background",
        "noisy-student",
        "advprop",
        "imagenet+5k",
    ]

    model_params = {
        "model_name": None,
        "encoder_name": None,
        "encoder_weights": None,
    }
    for model in models:
        if model + "_" in model_path:
            model_params["model_name"] = model
            break

    model_path = model_path.replace(model_params["model_name"] + "_", "*")

    for encoder in encoders:
        if "*" + encoder + "_" in model_path:
            model_params["encoder_name"] = encoder
            break

    for weight in weights:
        if "_" + weight + "_" in model_path:
            model_params["encoder_weights"] = weight
            break

    return model_params


def normalize_image(
    image: np.ndarray,
    target_min: Union[int, float] = 0.0,
    target_max: Union[int, float] = 1.0,
    target_type=np.float32,
) -> Union[int, float]:
    a = (target_max - target_min) / (image.max() - image.min())
    b = target_max - a * image.max()
    image_norm = (a * image + b).astype(target_type)
    return image_norm


class BorderExtractor:
    def __init__(
        self,
        thresh_method: str,
        thresh_val: int,
    ) -> None:

        self.thresh_method = thresh_method
        self.thresh_val = thresh_val
        assert self.thresh_method in ['otsu', 'triangle', 'manual'], f'Invalid thresh_method: {self.thresh_method}'

        if thresh_method == 'manual' and not isinstance(thresh_val, int):
            raise ValueError(f'Manual thresholding requires a thresholding value to be set. The thresh_val is {thresh_val}')

    def binarize(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:

        mask_bin = mask.copy()
        if self.thresh_method == 'otsu':
            thresh_value, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.thresh_method == 'triangle':
            thresh_value, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        elif self.thresh_method == 'manual':
            thresh_value, mask_bin = cv2.threshold(mask, self.thresh_val, 255, cv2.THRESH_BINARY)
        else:
            logging.warning(f'Invalid threshold: {self.thresh_val}')

        return mask_bin

    @staticmethod
    def extract_boundary(
        mask: np.ndarray,
    ) -> np.ndarray:
        _mask = Image.fromarray(mask)
        _mask = _mask.filter(ImageFilter.ModeFilter(size=7))
        _mask = np.asarray(_mask)
        mask_border = cv2.Canny(image=_mask, threshold1=100, threshold2=200)
        return mask_border

    @staticmethod
    def overlay_mask(
        image: np.ndarray,
        mask: np.ndarray,
        output_size: Tuple[int, int] = (1024, 1024),
        color: Tuple[int, int, int] = (255, 255, 0),
    ) -> np.ndarray:

        mask = cv2.resize(mask, dsize=output_size, interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, dsize=output_size, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image[mask == 255] = color

        return image