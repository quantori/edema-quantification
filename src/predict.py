import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.edema_net import EdemaNet
from src.models.feature_detector import FeatureDetector
from src.models.lung_segmenter import LungSegmenter

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
                batch_size=1,
                conf_threshold=0.01,
                device='auto',
            ),
        )

    EdemaNet(
        lung_segmenters=lung_segmenters,
        feature_detectors=feature_detectors,
        nms_method='soft',
        iou_threshold=0.5,
        conf_threshold=0.7,
        output_size=(1536, 1536),
        lung_extension=(50, 50, 50, 150),
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
