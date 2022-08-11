import os
import cv2
import logging
import argparse
from pathlib import Path

from tqdm import tqdm
from tools.models import LungSegmentation
from tools.utils import MorphologicalTransformations, get_file_list

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
        output_dir: str,
        thresh_method: str = 'otsu',
        thresh_val: float = 0.5,
) -> None:

    logger.info(f'Settings..................:')
    logger.info(f'Image directory...........: {img_dir}')
    logger.info(f'Output directory..........: {output_dir}')
    logger.info(f'Threshold method..........: {thresh_method.capitalize()}')
    logger.info(f'Threshold value...........: {thresh_val}')

    model = LungSegmentation(
        model_dir=model_dir,
        threshold=0.50,
        device='auto',
        raw_output=True,
    )

    img_paths = get_file_list(
        src_dirs=img_dir,
        ext_list=[
            '.png',
            '.jpg',
            '.jpeg',
            '.bmp',
        ]
    )

    for img_path in tqdm(img_paths, desc='Boundary extraction', unit=' images'):
        # img_path = img_dir + '/' + img_path
        mask = model(img_path)
        cv2.imwrite(f'dataset/output/masks/mask_{img_path}.png', mask)
        morph = MorphologicalTransformations(
            image_file=f'dataset/output/masks/mask_{img_path}.png'
        )

        binarized_mask = morph.binary(thresh_method, thresh_val)
        boundary = morph.extract_boundary(binarized_mask)
        image_bound = morph.visualize_boundary(img_path, boundary)
        filename = os.path.split(img_path)[-1]
        cv2.imwrite(os.path.join(output_dir, filename), image_bound)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Boundary extraction')
    parser.add_argument('--img_dir', default='dataset/img', type=str)
    parser.add_argument('--model_dir', default='models/lung_segmentation/DeepLabV3+', type=str)
    parser.add_argument('--threshold_method', default='otsu', type=str, choices=['otsu', 'triangle', 'manual'])
    parser.add_argument('--threshold_value', type=float, default=None)
    parser.add_argument('--output_dir', default='dataset/output/boundary', type=str)
    args = parser.parse_args()

    boundary_extraction(
        img_dir=args.img_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        thresh_method=args.threshold_method,
        thresh_val=args.threshold_value,
    )
