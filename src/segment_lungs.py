import logging
import os
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import get_file_list
from src.models.models import BorderExtractor, LungSegmentation

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='segment_lungs',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    if cfg.save_dir is None:
        model_name = Path(cfg.model_dir).name
        cfg.save_dir = f'{cfg.img_dir}_{model_name}'

    map_dir = os.path.join(cfg.save_dir, 'map')
    mask_dir = os.path.join(cfg.save_dir, 'mask')
    border_dir = os.path.join(cfg.save_dir, 'delineation')
    os.makedirs(map_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(border_dir, exist_ok=True)

    log.info(f'Settings..................:')
    log.info(f'Image directory...........: {cfg.img_dir}')
    log.info(f'Output directory..........: {cfg.save_dir}')
    log.info(f'Output size...............: {cfg.output_size}')
    log.info(f'Threshold method..........: {cfg.threshold_method.capitalize()}')
    log.info(f'Threshold value...........: {cfg.threshold_val}')

    img_paths = get_file_list(
        src_dirs=cfg.img_dir,
        ext_list=[
            '.png',
            '.jpg',
            '.jpeg',
            '.bmp',
        ],
    )
    log.info(f'Number of images..........: {len(img_paths)}')

    model = LungSegmentation(
        model_dir=cfg.model_dir,
        threshold=0.50,
        device='auto',
        raw_output=True,
    )

    extractor = BorderExtractor(
        threshold_method=cfg.threshold_method,
        threshold_val=cfg.threshold_val,
    )

    for img_path in tqdm(img_paths, desc='Lung segmentation', unit='images'):
        img_name = Path(img_path).name
        img_input = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Get and save a probability map
        map = model(img_path)
        map_path = os.path.join(map_dir, img_name)
        cv2.imwrite(map_path, map)

        # Get and save a binary mask
        mask = extractor.binarize(mask=map)
        mask_border = extractor.extract_boundary(
            mask=mask,
        )
        mask_path = os.path.join(mask_dir, img_name)
        cv2.imwrite(mask_path, mask)

        # Get and save a delineated image
        img_output = extractor.overlay_mask(
            image=img_input,
            mask=mask_border,
            output_size=cfg.output_size,
            color=(255, 255, 0),
        )
        img_output_path = os.path.join(border_dir, img_name)
        cv2.imwrite(img_output_path, img_output)


if __name__ == '__main__':
    main()
