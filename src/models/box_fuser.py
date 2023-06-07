from pathlib import Path

import numpy as np
import pandas as pd
from ensemble_boxes import nms, soft_nms
from tqdm import tqdm

from src.data.utils_sly import FEATURE_MAP_REVERSED


class BoxFuser:
    """BoxFuser is a class for fusing multiple boxes."""

    def __init__(
        self,
        method: str = 'soft_nms',
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
    ):
        assert 0 <= iou_threshold <= 1, 'iou_threshold must lie within [0, 1]'
        assert 0 <= conf_threshold <= 1, 'conf_threshold must lie within [0, 1]'
        assert method in ['nms', 'soft_nms', 'nmw', 'wbf'], f'Unknown fusion method: {method}'
        self.method = method
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def fuse_detections(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df.drop(columns=['ID'], inplace=True)
        df_out = pd.DataFrame(columns=df.columns)

        # Process predictions one image at a time
        img_groups = df.groupby('Image path')
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

            if self.method == 'nms':
                boxes, scores, labels = nms(
                    boxes=box_list,
                    scores=score_list,
                    labels=label_list,
                    iou_thr=self.iou_threshold,
                )
            elif self.method == 'soft_nms':
                boxes, scores, labels = soft_nms(
                    boxes=box_list,
                    scores=score_list,
                    labels=label_list,
                    iou_thr=self.iou_threshold,
                    method=2,
                    sigma=0.1,
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

        # TODO: Think of the order: conf_threshold and nms, or vice versa
        df_out = df_out[df_out['Confidence'] >= self.conf_threshold]

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
    # Create an instance of BoxFuser
    box_fuser = BoxFuser(
        method='soft_nms',
        iou_threshold=0.5,
        conf_threshold=0.01,
    )

    # Suppress and/or fuse boxes
    dets = pd.read_excel('data/coco/test_demo/predictions.xlsx')
    dets_fused = box_fuser.fuse_detections(df=dets)
    print('Complete')
