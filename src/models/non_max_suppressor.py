from pathlib import Path

import numpy as np
import pandas as pd
from ensemble_boxes import nms, soft_nms
from tqdm import tqdm

from src.data.utils_sly import FEATURE_MAP_REVERSED


class NonMaxSuppressor:
    """NonMaxSuppressor is a class for fusing multiple boxes."""

    def __init__(
        self,
        conf_thresholds: dict,
        method: str = 'soft',
        sigma: float = 0.1,
        iou_threshold: float = 0.5,
    ):
        assert 0 <= iou_threshold <= 1, 'iou_threshold must lie within [0, 1]'
        for conf_threshold in conf_thresholds.values():
            assert 0 <= conf_threshold <= 1, 'conf_threshold must lie within [0, 1]'
        assert method in ['standard', 'soft'], f'Unknown fusion method: {method}'
        self.method = method
        self.iou_threshold = iou_threshold
        self.conf_thresholds = conf_thresholds
        self.sigma = sigma

    def suppress_detections(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        # If there are no detections, return an initial 1-row dataframe
        if len(df) == 1 and pd.isna(df.loc[0, 'Feature']):
            return df

        # Filter dataframe by feature confidence
        df_filtered = pd.DataFrame()
        for key, threshold in self.conf_thresholds.items():
            mask = (df['Feature'] == key) & (df['Confidence'] >= threshold)
            df_filtered = df_filtered.append(df[mask])

        # Process predictions one image at a time
        df_out = pd.DataFrame(columns=df_filtered.columns)
        img_groups = df_filtered.groupby('Image path')
        for img_id, (img_path, df_img) in tqdm(
            enumerate(img_groups),
            desc='Suppress detections',
            unit=' images',
            total=len(img_groups),
        ):
            # Get normalized list of box coordinates
            box_list = []
            for _, row in df_img.iterrows():
                box_list.append(
                    [
                        [
                            row['x1'] / row['Image width'],
                            row['y1'] / row['Image height'],
                            row['x2'] / row['Image width'],
                            row['y2'] / row['Image height'],
                        ],
                    ],
                )

            # Get list of confidence scores
            score_list = [df_img['Confidence'].values.tolist()]

            # Get list of box labels
            label_list = [df_img['Feature ID'].values.tolist()]

            if self.method == 'standard':
                boxes, scores, labels = nms(
                    boxes=box_list,
                    scores=score_list,
                    labels=label_list,
                    iou_thr=self.iou_threshold,
                )
            elif self.method == 'soft':
                boxes, scores, labels = soft_nms(
                    boxes=box_list,
                    scores=score_list,
                    labels=label_list,
                    iou_thr=self.iou_threshold,
                    sigma=self.sigma,
                )
            else:
                raise ValueError(f'Unknown method: {self.method}')

            # Convert box coordinates back and compute box metadata
            df_img_ = self.convert_fused_detections(
                df=df_img,
                boxes=boxes,
                scores=scores,
                labels=labels,
            )

            df_out = pd.concat([df_out, df_img_])

        return df_out

    @staticmethod
    def convert_fused_detections(
        df: pd.DataFrame,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> pd.DataFrame:
        df_out = pd.DataFrame(index=range(len(boxes)), columns=df.columns)
        img_width = df['Image width'].unique()[0]
        img_height = df['Image height'].unique()[0]
        img_path = df['Image path'].unique()[0]
        img_name = Path(img_path).name
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
        df_out['Feature ID'] = labels
        df_out['Feature'] = df_out.apply(
            func=lambda row: FEATURE_MAP_REVERSED[row['Feature ID']],
            axis=1,
        )
        df_out['Confidence'] = scores

        return df_out


if __name__ == '__main__':
    # Create an instance of NonMaxSuppressor
    import os

    test_dir = 'data/coco/test'

    conf_thresholds = dict(
        Cephalization=0.2,
        Artery=0.2,
        Heart=0.2,
        Kerley=0.2,
        Bronchus=0.2,
        Effusion=0.2,
        Bat=0.2,
        Infiltrate=0.2,
        Cuffing=0.2,
        Lungs=0.2,
    )

    box_fuser = NonMaxSuppressor(
        method='soft',
        sigma=0.1,
        iou_threshold=0.5,
        conf_thresholds=conf_thresholds,
    )

    # Suppress and/or fuse boxes
    df_dets = pd.read_excel(os.path.join(test_dir, 'predictions.xlsx'))
    df_dets_fused = box_fuser.suppress_detections(df=df_dets)
    df_dets_fused.drop(columns=['ID'], inplace=True)
    df_dets_fused.index += 1
    df_dets_fused.to_excel(
        os.path.join(test_dir, 'predictions_nms.xlsx'),
        sheet_name='Detections',
        index=True,
        index_label='ID',
    )
    print('Complete')
