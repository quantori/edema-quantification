import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from mmdet.apis import inference_detector, init_detector

from src.data.utils import get_file_list
from src.data.utils_sly import FEATURE_MAP


class FeatureDetector:
    """A class used for the detection of radiological features."""

    def __init__(
        self,
        model_dir: str,
        conf_threshold: float = 0.01,
        iou_threshold: float = 0.5,
        device: str = 'auto',
    ):
        # Get config path
        config_list = get_file_list(
            src_dirs=model_dir,
            ext_list='.py',
        )
        assert len(config_list) == 1, 'Keep only one config file in the model directory'
        config_path = config_list[0]

        # Get checkpoint path
        checkpoint_list = get_file_list(
            src_dirs=model_dir,
            ext_list='.pth',
        )
        assert len(checkpoint_list) == 1, 'Keep only one checkpoint file in the model directory'
        checkpoint_path = checkpoint_list[0]

        # Load the model
        if device == 'cpu':
            device_ = 'cpu'
        elif device == 'gpu':
            device_ = 'cuda'
        elif device == 'auto':
            device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            raise ValueError(f'Unknown device: {device}')

        self.model = init_detector(
            config=config_path,
            checkpoint=checkpoint_path,
            device=device_,
        )
        self.features = self.model.CLASSES

        # Set conf_threshold
        try:
            self.model.test_cfg.rcnn.score_thr = conf_threshold
        except Exception:
            self.model.test_cfg.score_thr = conf_threshold

        # Set iou_threshold
        if 'nms' in self.model.test_cfg:
            self.model.test_cfg.nms.iou_threshold = iou_threshold
        elif 'rpn' in self.model.test_cfg and 'rcnn' in self.model.test_cfg:
            self.model.test_cfg.rpn.nms.iou_threshold = iou_threshold
            self.model.test_cfg.rcnn.nms.iou_threshold = iou_threshold
        else:
            raise ValueError('Unknown case for the assignment of iou_threshold')

        # Log model parameters
        logging.info('')
        logging.info(f'Model.....................: {self.model.cfg.model["type"]}')
        logging.info(f'Model dir.................: {model_dir}')
        logging.info(f'Confidence threshold......: {conf_threshold}')

    def predict(
        self,
        img: np.ndarray,
    ) -> List[np.ndarray]:
        detections = inference_detector(model=self.model, imgs=img)

        return detections

    def process_detections(
        self,
        img_path: str,
        detections: List[np.ndarray],
    ) -> pd.DataFrame:
        columns = [
            'Image path',
            'Image name',
            'Image height',
            'Image width',
            'x1',
            'y1',
            'x2',
            'y2',
            'Box width',
            'Box height',
            'Box area',
            'Feature ID',
            'Feature',
            'Confidence',
        ]

        df = pd.DataFrame(columns=columns)
        img_height, img_width = cv2.imread(img_path).shape[:2]

        # Return a 1-row dataframe if there are no detections
        if all(len(arr) == 0 for arr in detections):
            row_data = {
                'Image path': [img_path],
                'Image name': [Path(img_path).name],
                'Image height': [img_height],
                'Image width': [img_width],
            }
            df = pd.DataFrame(data=row_data, columns=columns)
            return df

        # Iterate over class detections
        for feature_idx, feature_detections in enumerate(detections):
            if feature_detections.size == 0:
                num_detections = 1
            else:
                num_detections = feature_detections.shape[0]

            # Iterate over boxes on a single image
            df_ = pd.DataFrame(index=range(num_detections), columns=columns)
            df_['Image path'] = img_path
            df_['Image name'] = Path(img_path).name
            df_['Image height'] = img_height
            df_['Image width'] = img_width
            for box_idx, box in enumerate(feature_detections):
                # box -> array(x_min, y_min, x_max, y_max, confidence)
                df_.at[box_idx, 'x1'] = int(box[0])
                df_.at[box_idx, 'y1'] = int(box[1])
                df_.at[box_idx, 'x2'] = int(box[2])
                df_.at[box_idx, 'y2'] = int(box[3])
                df_.at[box_idx, 'Feature'] = self.features[feature_idx]
                df_.at[box_idx, 'Feature ID'] = FEATURE_MAP[self.features[feature_idx]]
                df_.at[box_idx, 'Confidence'] = box[4]

            df = pd.concat([df, df_])

        df['Box width'] = abs(df.x2 - df.x1 + 1)
        df['Box height'] = abs(df.y2 - df.y1 + 1)
        df['Box area'] = df['Box width'] * df['Box height']

        df.dropna(subset=['x1', 'x2', 'y1', 'y2', 'Confidence'], inplace=True)
        df.sort_values('Feature ID', inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df


if __name__ == '__main__':
    import os

    test_dir = 'data/coco/test/'
    img_paths = get_file_list(
        src_dirs=os.path.join(test_dir, 'data'),
        ext_list='.png',
    )
    model = FeatureDetector(
        model_dir='models/feature_detection/FasterRCNN_ResNet50',
        conf_threshold=0.01,
        iou_threshold=0.5,
        device='auto',
    )
    df_dets = pd.DataFrame()
    for img_path in img_paths:
        img = cv2.imread(img_path)
        dets = model.predict(img)
        df_dets_ = model.process_detections(
            img_path=img_path,
            detections=dets,
        )
        df_dets = pd.concat([df_dets, df_dets_])
    df_dets.index += 1
    df_dets.to_excel(
        os.path.join(test_dir, 'predictions.xlsx'),
        sheet_name='Detections',
        index=True,
        index_label='ID',
    )
