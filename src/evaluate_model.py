import os
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, List, Sequence, Tuple, Union

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.object_detection_metrics.lib.BoundingBox import BoundingBox
from src.object_detection_metrics.lib.BoundingBoxes import BoundingBoxes
from src.object_detection_metrics.lib.Evaluator import Evaluator
from src.object_detection_metrics.lib.utils import BBFormat, BBType, CoordinatesType


def _get_confidence_array(min_value: float, max_value: float, step: float) -> np.ndarray:
    return np.fromiter((i for i in np.arange(min_value, max_value + step, step)), dtype=float)


def _get_bounding_boxes(
    df_gt: pd.DataFrame,
    df_pred: pd.DataFrame,
    confidence_threshold: float,
) -> BoundingBoxes:
    bounding_boxes = BoundingBoxes()
    bounding_boxes = _add_gt_bboxes(df_gt, bounding_boxes)
    df_pred_conf_filtered = df_pred[df_pred['Confidence'] >= confidence_threshold]
    bounding_boxes = _add_pred_bboxes(df_pred_conf_filtered, bounding_boxes)
    return bounding_boxes


def _add_gt_bboxes(df_gt: pd.DataFrame, bounding_boxes: BoundingBoxes) -> BoundingBoxes:
    for row in df_gt.itertuples():
        bb_gt = BoundingBox(
            imageName=row._3,
            classId=row.Feature,
            x=row.x1,
            y=row.y1,
            w=row.x2,
            h=row.y2,
            typeCoordinates=CoordinatesType.Absolute,
            bbType=BBType.GroundTruth,
            format=BBFormat.XYX2Y2,
        )
        bounding_boxes.addBoundingBox(bb_gt)
    return bounding_boxes


def _add_pred_bboxes(df_pred: pd.DataFrame, bounding_boxes: BoundingBoxes) -> BoundingBoxes:
    for row in df_pred.itertuples():
        bb_pred = BoundingBox(
            imageName=row._3,
            classId=row.Feature,
            classConfidence=row.Confidence,
            x=row.x1,
            y=row.y1,
            w=row.x2,
            h=row.y2,
            typeCoordinates=CoordinatesType.Absolute,
            bbType=BBType.Detected,
            format=BBFormat.XYX2Y2,
        )
        bounding_boxes.addBoundingBox(bb_pred)
    return bounding_boxes


def evaluate(
    confidence_threshold: float,
    df_gt: pd.DataFrame,
    df_pred: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> Dict[str, Union[float, List[Dict[str, Any]]]]:
    bounding_boxes = _get_bounding_boxes(df_gt, df_pred, confidence_threshold)
    evaluator = Evaluator()
    metrics = evaluator.GetPascalVOCMetrics(bounding_boxes, IOUThreshold=iou_threshold)
    return {'confidence_threshold': confidence_threshold, 'metrics': metrics}


def _exclude_features(
    dfs: Tuple[pd.DataFrame, ...],
    features: Sequence[str],
) -> Union[List[pd.DataFrame], Tuple[pd.DataFrame, ...]]:
    # The order of DataFrames in dfs has to match the order of DataFrames to which the return values
    # are assigned.
    if len(features) > 0:
        dfs_output = []
        for df in dfs:
            dfs_output.append(df[~df['Feature'].isin(features)])
        return dfs_output
    else:
        return dfs


def _create_df(results: List[Dict[str, Union[float, List[Dict[str, Any]]]]]) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=['Class', 'AP', 'Total positives', 'Total TP', 'Total FP', 'Confidence'],
    )
    for result in results:
        for cls in result['metrics']:  # type: ignore
            df = df.append(
                {
                    'Class': cls['class'],
                    'AP': cls['AP'],
                    'Total positives': cls['total positives'],
                    'Total TP': cls['total TP'],
                    'Total FP': cls['total FP'],
                    'Confidence': result['confidence_threshold'],
                },
                ignore_index=True,
            )
    return df


def _save_df(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, 'detection_metrics.xlsx')
    df.to_excel(
        metrics_path,
        sheet_name='Metrics',
    )


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='evaluate_model',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Get list of possible confidence thresholds.
    conf_thresholds = _get_confidence_array(
        min_value=cfg.conf_range[0],
        max_value=cfg.conf_range[1],
        step=cfg.conf_step,
    )

    # Read DataFrames and exclude features.
    df_gt = pd.read_excel(cfg.gt_path)
    df_pred = pd.read_excel(cfg.pred_path)
    # TODO: Clarify if df_pred is supposed to have 'Heart', 'Lungs', etc.
    df_gt_filtered, df_pred_filterd = _exclude_features((df_gt, df_pred), cfg.exclude_features)

    # Compute metrics for all confidence thresholds classwise.
    with Pool() as p:
        results = list(
            tqdm(
                p.imap(
                    partial(
                        evaluate,
                        df_gt=df_gt_filtered,
                        df_pred=df_pred_filterd,
                        iou_threshold=cfg.iou_threshold,
                    ),
                    conf_thresholds,
                ),
                total=len(conf_thresholds),
            ),
        )

    # Create and save a DataFrame with the metrics.
    df = _create_df(results)
    _save_df(df, cfg.save_dir)


if __name__ == '__main__':
    main()
