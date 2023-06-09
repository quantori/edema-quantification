import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.model_evaluator import ModelEvaluator

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='evaluate_model',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Initialize evaluator instance
    evaluator = ModelEvaluator(
        iou_threshold=cfg.iou_threshold,
        conf_threshold=cfg.conf_threshold,
    )

    # Combine ground truth and predictions
    dets = evaluator.combine_data(
        gt_path=cfg.gt_path,
        pred_path=cfg.pred_path,
        exclude_features=cfg.exclude_features,
    )

    # Estimate key detection metrics
    if cfg.is_evaluate:
        tp, fp, fn = evaluator.evaluate(detections=dets)
        f1_score = tp / (tp + 0.5 * (fp + fn))
        log.info(f'TP: {tp}')
        log.info(f'FP: {fp}')
        log.info(f'FN: {fn}')
        log.info(f'F1: {f1_score:.1%}')

    # Visually compare ground truth and predictions
    if cfg.is_visualize:
        evaluator.visualize(detections=dets)

    log.info('Complete')


if __name__ == '__main__':
    main()
