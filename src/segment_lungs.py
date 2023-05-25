import gc
import logging
import os
from pathlib import Path

import cv2
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import get_file_list
from src.models.lung_segmenter import LungSegmenter

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='segment_lungs',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

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
    img_paths = img_paths[:10]  # TODO: remove after debugging

    for model_name in cfg.model_names:
        print(f'{model_name} model is in use')
        model_dir = os.path.join(cfg.model_dirs, model_name)
        map_dir = os.path.join(cfg.save_dir, f'{model_name}', 'map')
        mask_dir = os.path.join(cfg.save_dir, f'{model_name}', 'mask')
        os.makedirs(map_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        model = LungSegmenter(
            model_dir=model_dir,
            device='auto',
        )

        for img_path in tqdm(img_paths, desc='Lung segmentation', unit='images'):
            img_name = Path(img_path).name
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]

            # Retrieve and save a probability segmentation map
            map_ = model(
                img=img,
                scale_output=True,
            )
            map = cv2.resize(map_, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
            map_path = os.path.join(map_dir, img_name)
            cv2.imwrite(map_path, map)

            # Retrieve and save a binary segmentation mask
            mask_ = model.binarize_map(
                map=map,
                threshold_method='otsu',
            )
            mask_smooth = model.smooth_mask(mask=mask_)
            mask_lungs = model.keep_lungs(mask=mask_smooth)
            mask_path = os.path.join(mask_dir, img_name)
            cv2.imwrite(mask_path, mask_lungs)

            # TODO: Compute lung coordinates
            model.compute_lung_coords(mask=mask_lungs)

        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
