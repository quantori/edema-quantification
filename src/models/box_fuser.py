from typing import List, Optional, Union

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

from src.data.utils_sly import FEATURE_MAP_REVERSED


class BoxFuser:
    """A class for fusing bounding boxes from a few models."""

    def __init__(
        self,
        weights: Optional[List[int]] = None,
        iou_threshold: float = 0.5,
        skip_box_threshold: float = 0.0,
        conf_type: str = 'avg',
        allows_overflow: bool = False,
    ) -> None:
        self.weights = weights
        self.iou_thr = iou_threshold
        self.skip_box_thr = skip_box_threshold
        self.conf_type = conf_type
        self.allows_overflow = allows_overflow

    def fuse_detections(
        self,
        df_list: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """The main fusing function.

        Args:
            df_list: a container with DataFrames, where bounding boxes have to be fused.

        Returns:
            df_out: a DataFrame with fused bounding boxes.
        """
        # No box fusion is required for one model
        if len(df_list) == 1:
            return df_list[0]

        box_list: List[Union[List[float], List[List[float]]]] = []
        score_list: List[Union[float, List[float]]] = []
        label_list: List[Union[int, List[int]]] = []

        for df in df_list:
            box_list.append(BoxFuser._get_normalized_boxes(df))
            score_list.append(BoxFuser._get_scores(df))
            label_list.append(BoxFuser._get_labels(df))

        # Fuse detections.
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list=box_list,
            scores_list=score_list,
            labels_list=label_list,
            weights=self.weights,
            iou_thr=self.iou_thr,
            skip_box_thr=self.skip_box_thr,
            conf_type=self.conf_type,
            allows_overflow=self.allows_overflow,
        )

        # Convert box coordinates back and compute box metadata.
        df_out = BoxFuser._convert_fused_detections(
            df=df,
            boxes=boxes,
            scores=scores,
            labels=labels,
        )

        return df_out

    @staticmethod
    def _get_normalized_boxes(df: pd.DataFrame) -> List[List[float]]:
        bboxes: List[List[float]] = []
        for _, row in df.iterrows():
            bboxes.append(
                [
                    row['x1'] / row['Image width'],
                    row['y1'] / row['Image height'],
                    row['x2'] / row['Image width'],
                    row['y2'] / row['Image height'],
                ],
            )
        return bboxes

    @staticmethod
    def _get_scores(df: pd.DataFrame) -> List[float]:
        return df['Confidence'].values.tolist()

    @staticmethod
    def _get_labels(df: pd.DataFrame) -> List[int]:
        return df['Feature ID'].values.tolist()

    @staticmethod
    def _convert_fused_detections(
        df: pd.DataFrame,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> pd.DataFrame:
        df_out = pd.DataFrame(index=range(len(boxes)), columns=df.columns)
        img_width = df['Image width'].unique()[0]
        img_height = df['Image height'].unique()[0]
        img_path = df['Image path'].unique()[0]
        img_name = df['Image name'].unique()[0]
        df_out['Image path'] = img_path
        df_out['Image name'] = img_name
        df_out['Image height'] = img_height
        df_out['Image width'] = img_width
        df_out['x1'] = (boxes[:, 0] * img_width).astype(int)
        df_out['y1'] = (boxes[:, 1] * img_width).astype(int)
        df_out['x2'] = (boxes[:, 2] * img_width).astype(int)
        df_out['y2'] = (boxes[:, 3] * img_width).astype(int)
        df_out['Box width'] = df_out.apply(
            func=lambda row: abs(row['x2'] - row['x1'] + 1),
            axis=1,
        )
        df_out['Box height'] = df_out.apply(
            func=lambda row: abs(row['y2'] - row['y1'] + 1),
            axis=1,
        )
        df_out['Box area'] = df_out.apply(
            func=lambda row: row['Box width'] * row['Box height'],
            axis=1,
        )
        df_out['Feature ID'] = labels.astype(int)
        df_out['Feature'] = df_out.apply(
            func=lambda row: FEATURE_MAP_REVERSED[row['Feature ID']],
            axis=1,
        )
        df_out['Confidence'] = scores

        return df_out


if __name__ == '__main__':
    df = pd.read_excel('./data/coco/test/predictions2.xlsx')
    # df = pd.DataFrame(columns=METADATA_COLUMNS)
    dfs = [df.head(7), df.head(7), df.head(7)]
    fuser = BoxFuser()
    df_o = fuser.fuse_detections(dfs)
    print(df_o)
