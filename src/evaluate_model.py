import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from src import BBFormat, BBType, BoundingBox, BoundingBoxes, CoordinatesType, Evaluator


def _get_confidence_array(
    min_value: float,
    max_value: float,
    step: float,
) -> np.ndarray:
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


def _add_gt_bboxes(
    df_gt: pd.DataFrame,
    bounding_boxes: BoundingBoxes,
) -> BoundingBoxes:
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


def _add_pred_bboxes(
    df_pred: pd.DataFrame,
    bounding_boxes: BoundingBoxes,
) -> BoundingBoxes:
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


def _create_df(
    results: List[Dict[str, Union[float, List[Dict[str, Any]]]]],
) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            'Class',
            'AP',
            'Total positives',
            'Total TP',
            'Total FP',
            'Total FN',
            'Precision',
            'Recall',
            'F1',
            'F0.5',
            'F2',
            'Confidence',
        ],
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
                    'Total FN': calculate_false_negatives(cls['recall'], cls['total TP']),
                    'Precision': cls['precision'][-1] if cls['precision'].size != 0 else 0,
                    'Recall': cls['recall'][-1] if cls['recall'].size != 0 else 0,
                    'F1': calculate_f_beta(cls['precision'], cls['recall']),
                    'F0.5': calculate_f_beta(cls['precision'], cls['recall'], beta=0.5),
                    'F2': calculate_f_beta(cls['precision'], cls['recall'], beta=2),
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
    df.index += 1
    df.to_excel(
        metrics_path,
        sheet_name='Metrics',
        index=True,
        index_label='ID',
    )


def calculate_f_beta(
    precision: np.ndarray,
    recall: np.ndarray,
    beta: float = 1,
) -> float:
    if (precision.size != 0 and recall.size != 0) and (precision[-1] != 0 or recall[-1] != 0):
        return (1 + beta**2) * (
            (precision[-1] * recall[-1]) / ((beta**2 * precision[-1]) + recall[-1])
        )
    else:
        return 0.0


def calculate_false_negatives(
    recall: np.ndarray,
    total_tp: float,
) -> int:
    if recall.size != 0 and recall[-1] != 0:
        return round(total_tp / recall[-1] - total_tp)
    else:
        return 0


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
    df_gt_filtered, df_pred_filterd = _exclude_features((df_gt, df_pred), cfg.exclude_features)

    # Compute metrics for all confidence thresholds class-wise
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(evaluate)(
            confidence_threshold=conf_threshold,
            df_gt=df_gt_filtered,
            df_pred=df_pred_filterd,
            iou_threshold=cfg.iou_threshold,
        )
        for conf_threshold in tqdm(conf_thresholds, desc='Model evaluation')
    )

    # Create and save a DataFrame with the metrics.
    df = _create_df(results)
    _save_df(df, cfg.save_dir)


if __name__ == '__main__':
    main()
