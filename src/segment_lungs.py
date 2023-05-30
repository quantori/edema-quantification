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

    for model_name in cfg.model_names:
        log.info(f'Model in use: {model_name}')
        print(f'Model in use: {model_name}')
        model_dir = os.path.join(cfg.model_dirs, model_name)
        img_dir = os.path.join(cfg.save_dir, f'{model_name}')
        os.makedirs(img_dir, exist_ok=True)

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
            map_path = os.path.join(img_dir, img_name)
            cv2.imwrite(map_path, map)

        # Run the garbage collector and release all unused cached memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    log.info('Complete')


if __name__ == '__main__':
    main()
