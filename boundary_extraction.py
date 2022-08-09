import os
import logging
import argparse
import cv2
from pathlib import Path
from typing import Union
from tools.models import LungSegmentation
from tools.utils import MorphologicalTransformations

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
        folder_dir: str,
        output_dir: str,
        model_name: str,
        thresh_type:str,
        thresh_val
) -> None:

    model = LungSegmentation(
        model_dir=f'models/lung_segmentation/{model_name}',
        threshold=0.50,
        device='auto',
        raw_output=True,
    )

    for images in os.listdir(folder_dir):
        # check if the image ends with png or jpg or jpeg
        if images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg"):
            img_path = folder_dir + '/' + images
            mask_lungs = model(img_path)  # segmented masks
            cv2.imwrite(f'dataset/output/masks/mask_{images}.png', mask_lungs)
        morph = MorphologicalTransformations(
            image_file=f'dataset/output/masks/mask_{images}.png'
        )
        """logger.info(f'Settings..................:')
        logger.info(f'folder dir.................: {folder_dir}')
        logger.info(f'output_dir ................: {output_dir}')
        logger.info(f'Model name................: {model_name}')
        logger.info(f'Threshold.................: {thresh_type}')"""

        binarized_mask = morph.binary(thresh_type, thresh_val) #takes both threshold value and thresh type
        boundary = morph.extract_boundary(binarized_mask)
        image_bound = morph.visualize_boundary(img_path, boundary)
        filename = os.path.split(images)[-1]
        cv2.imwrite(os.path.join(output_dir,filename),image_bound)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("boundary extraction")
    parser.add_argument("--dataset_dir", type=str, default="dataset/data")
    parser.add_argument("--output_dir", type=str, default="dataset/output/boundary")
    parser.add_argument("--model_name", type=str, default="Unet++")
    parser.add_argument("--threshold_type", type=str, default="Triangle")
    parser.add_argument("--threshold_value", type=float, default=None)
    args = parser.parse_args()

    boundary_extraction(args.dataset_dir, args.output_dir, args.model_name, args.threshold_type, args.threshold_value)

