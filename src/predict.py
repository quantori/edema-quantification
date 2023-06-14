import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import get_file_list
from src.models.edema_net import EdemaNet
from src.models.feature_detector import FeatureDetector
from src.models.lung_segmenter import LungSegmenter
from src.models.map_fuser import MapFuser
from src.models.mask_processor import MaskProcessor

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='predict',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Initialize lung segmentation models
    lung_segmenters = []
    for model_dir in cfg.seg_model_dirs:
        lung_segmenters.append(
            LungSegmenter(
                model_dir=model_dir,
                device='auto',
            ),
        )

    # Initialize feature detection models
    feature_detectors = []
    for model_dir in cfg.det_model_dirs:
        feature_detectors.append(
            FeatureDetector(
                model_dir=model_dir,
                conf_threshold=0.01,
                device='auto',
            ),
        )

    # Initialize probability map fuser
    map_fuser = MapFuser()

    # Initialize binary mask processor
    mask_processor = MaskProcessor(
        threshold_method='otsu',
        kernel_size=(7, 7),
    )

    edema_net = EdemaNet(
        lung_segmenters=lung_segmenters,
        feature_detectors=feature_detectors,
        map_fuser=map_fuser,
        mask_processor=mask_processor,
        nms_method='soft',
        iou_threshold=0.5,
        conf_threshold=0.7,
        output_size=(1536, 1536),
        lung_extension=(50, 50, 50, 150),
    )

    # Get list of images to predict
    img_paths = get_file_list(
        src_dirs=cfg.data_dir,
        ext_list=[
            '.png',
            '.jpg',
            '.jpeg',
            '.bmp',
        ],
    )
    log.info(f'Number of images..........: {len(img_paths)}')
    for img_path in tqdm(img_paths, desc='Prediction', unit=' images'):
        log.info(f'Processing: {Path(img_path).stem}')
        edema_net.predict(img_path)

    log.info('Complete')


if __name__ == '__main__':
    main()
