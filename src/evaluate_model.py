import logging
import os
from typing import List, Tuple

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.models.model_evaluator import ModelEvaluator

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_threshold_list(
    min_value: float = 0.0,
    max_value: float = 1.0,
    step: float = 0.01,
) -> List[float]:
    threshold_list = []
    current_value = min_value
    while current_value <= max_value:
        threshold_list.append(current_value)
        current_value += step
        current_value = round(current_value, 2)

    return threshold_list


def process_threshold_pair(
    threshold_pair: Tuple[float, float],
    gt_path: str,
    pred_path: str,
    exclude_features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    iou_threshold, conf_threshold = threshold_pair

    # Initialize evaluator instance
    evaluator = ModelEvaluator(
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
    )

    # Combine ground truth and predictions
    detections = evaluator.combine_data(
        gt_path=gt_path,
        pred_path=pred_path,
        exclude_features=exclude_features,
    )

    # Estimate key detection metrics
    df_metrics, df_metrics_cw = evaluator.evaluate(detections=detections)

    return df_metrics, df_metrics_cw


def save_metrics(
    df_metrics: pd.DataFrame,
    df_metrics_cw: pd.DataFrame,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Save main metrics
    save_path = os.path.join(save_dir, 'metrics.xlsx')
    df_metrics.reset_index(drop=True, inplace=True)
    df_metrics.index += 1
    df_metrics.to_excel(
        save_path,
        sheet_name='Metrics',
        index=True,
        index_label='ID',
    )

    # Save class-wise metrics
    save_path = os.path.join(save_dir, 'metrics_cw.xlsx')
    df_metrics_cw.reset_index(drop=True, inplace=True)
    df_metrics_cw.index += 1
    df_metrics_cw.to_excel(
        save_path,
        sheet_name='Metrics',
        index=True,
        index_label='ID',
    )


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='evaluate_model',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Get list of possible IoU thresholds
    iou_thresholds = get_threshold_list(
        min_value=cfg.iou_range[0],
        max_value=cfg.iou_range[1],
        step=cfg.iou_step,
    )

    # Get list of possible confidence thresholds
    conf_thresholds = get_threshold_list(
        min_value=cfg.conf_range[0],
        max_value=cfg.conf_range[1],
        step=cfg.conf_step,
    )

    # Get list of possible threshold pair
    threshold_pairs = [(iou, conf) for iou in iou_thresholds for conf in conf_thresholds]

    # Compute metrics for all threshold pairs
    df_metrics = pd.DataFrame()
    df_metrics_cw = pd.DataFrame()
    for threshold_pair in tqdm(threshold_pairs, desc='Evaluation', unit='pair'):
        df_metrics_, df_metrics_cw_ = process_threshold_pair(
            threshold_pair=threshold_pair,
            gt_path=cfg.gt_path,
            pred_path=cfg.pred_path,
            exclude_features=cfg.exclude_features,
        )
        df_metrics = pd.concat([df_metrics, df_metrics_])
        df_metrics_cw = pd.concat([df_metrics_cw, df_metrics_cw_])

    # Save metrics
    save_metrics(
        df_metrics=df_metrics,
        df_metrics_cw=df_metrics_cw,
        save_dir=cfg.save_dir,
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
