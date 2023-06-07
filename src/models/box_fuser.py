from pathlib import Path

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
    ):
        assert 0 <= iou_threshold <= 1, 'iou_threshold must lie within [0, 1]'
        assert method in ['nms', 'soft_nms', 'nmw', 'wbf'], f'Unknown fusion method: {method}'
        self.method = method
        self.iou_threshold = iou_threshold

    def fuse_detections(
        self,
        detections: pd.DataFrame,
    ) -> pd.DataFrame:
        detections.drop(columns=['ID'], inplace=True)
        df_out = pd.DataFrame(columns=detections.columns)

        # Process predictions one image at a time
        img_groups = detections.groupby('Image path')
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
                    sigma=0.1,  # TODO: check the effect of this variable
                )
            else:
                raise ValueError(f'Unknown method: {self.method}')

            # TODO: move to another function
            df_img_ = pd.DataFrame(index=range(len(boxes)), columns=detections.columns)
            img_width = df_img['Image width'].unique()[0]
            img_height = df_img['Image height'].unique()[0]
            df_img_['Image path'] = img_path
            df_img_['Image name'] = Path(img_path).name
            df_img_['Image height'] = img_height
            df_img_['Image width'] = img_width
            df_img_['x1'] = (boxes[:, 0] * img_width).astype(int)
            df_img_['y1'] = (boxes[:, 1] * img_width).astype(int)
            df_img_['x2'] = (boxes[:, 2] * img_width).astype(int)
            df_img_['y2'] = (boxes[:, 3] * img_width).astype(int)
            df_img_['Box width'] = df_img_.apply(
                func=lambda row: abs(row['x2'] - row['x1'] + 1),
                axis=1,
            )
            df_img_['Box height'] = df_img_.apply(
                func=lambda row: abs(row['y2'] - row['y1'] + 1),
                axis=1,
            )
            df_img_['Box area'] = df_img_.apply(
                func=lambda row: row['Box width'] * row['Box height'],
                axis=1,
            )
            df_img_['Feature ID'] = labels
            df_img_['Feature'] = df_img_.apply(
                func=lambda row: FEATURE_MAP_REVERSED[row['Feature ID']],
                axis=1,
            )
            df_img_['Confidence'] = scores

        return df_out

    # def process_dataframe(
    #         self,
    #         df: pd.DataFrame,
    # ) -> pd.DataFrame:
    #
    #     # Convert boxes back
    #     return df_out


if __name__ == '__main__':
    # Create an instance of BoxFuser
    box_fuser = BoxFuser(
        method='soft_nms',
        iou_threshold=0.5,
    )

    # Suppress and/or fuse boxes
    dets = pd.read_excel('data/coco/test_demo/predictions.xlsx')
    dets_fused = box_fuser.fuse_detections(detections=dets)
    print('Complete')
