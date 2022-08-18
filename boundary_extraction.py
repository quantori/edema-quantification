import os
import cv2
import logging
import argparse
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
from tools.models import LungSegmentation
from tools.utils import BorderExtractor, get_file_list

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def boundary_extraction(
        img_dir: str,
        model_dir: str,
        save_dir: str,
        output_size: Tuple[int, int] = (1024, 1024),
        thresh_method: str = 'otsu',
        thresh_val: int = None,
) -> None:

    model_name = Path(model_dir).name
    mask_dir = os.path.join(save_dir, f'masks_{model_name}')
    border_dir = os.path.join(save_dir, f'border_{model_name}')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(border_dir, exist_ok=True)

    logger.info(f'Settings..................:')
    logger.info(f'Image directory...........: {img_dir}')
    logger.info(f'Output directory..........: {save_dir}')
    logger.info(f'Output size...............: {output_size}')
    logger.info(f'Threshold method..........: {thresh_method.capitalize()}')
    logger.info(f'Threshold value...........: {thresh_val}')

    img_paths = get_file_list(
        src_dirs=img_dir,
        ext_list=[
            '.png',
            '.jpg',
            '.jpeg',
            '.bmp',
        ],
    )
    logger.info(f'Number of images..........: {len(img_paths)}')

    model = LungSegmentation(
        model_dir=model_dir,
        threshold=0.50,
        device='auto',
        raw_output=True,
    )

    extractor = BorderExtractor(
        thresh_method=thresh_method,
        thresh_val=thresh_val,
    )

    for img_path in tqdm(img_paths, desc='Border extraction', unit=' images'):
        img_name = Path(img_path).name
        img_input = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        mask = model(img_path)
        mask_path = os.path.join(mask_dir, img_name)
        cv2.imwrite(mask_path, mask)

        mask_bin = extractor.binarize(mask=mask)
        mask_border = extractor.extract_boundary(
            mask=mask_bin,
        )
        img_output = extractor.overlay_mask(
            image=img_input,
            mask=mask_border,
            output_size=output_size,
            color=(255, 255, 0),
        )
        img_output_path = os.path.join(border_dir, img_name)
        cv2.imwrite(img_output_path, img_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Boundary extraction')
    parser.add_argument('--img_dir', default='dataset/img', type=str)
    parser.add_argument('--model_dir', default='models/lung_segmentation/DeepLabV3+', type=str)
    parser.add_argument('--output_size', default=(1024, 1024), nargs='+')
    parser.add_argument('--threshold_method', default='otsu', type=str, choices=['otsu', 'triangle', 'manual'])
    parser.add_argument('--threshold_value', type=int, default=None)
    parser.add_argument('--save_dir', default='dataset/output', type=str)
    args = parser.parse_args()

    boundary_extraction(
        img_dir=args.img_dir,
        model_dir=args.model_dir,
        save_dir=args.save_dir,
        output_size=tuple(args.output_size),
        thresh_method=args.threshold_method,
        thresh_val=args.threshold_value,
    )
