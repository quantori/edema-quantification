import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.detection_evaluator import DetectionEvaluator

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='visualize_predictions',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Initialize evaluator instance
    evaluator = DetectionEvaluator(
        iou_threshold=cfg.iou_threshold,
        conf_threshold=cfg.conf_threshold,
    )

    # Combine ground truth and predictions
    dets = evaluator.combine_data(
        gt_path=cfg.gt_path,
        pred_path=cfg.pred_path,
        exclude_features=cfg.exclude_features,
    )

    # Visually compare ground truth and predictions
    evaluator.visualize(detections=dets)

    log.info('Complete')


if __name__ == '__main__':
    main()
