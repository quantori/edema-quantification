import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from cpuinfo import get_cpu_info
from mmdet.apis import inference_detector, init_detector

from src.data.utils import get_file_list


class FeatureDetector:
    """A class used for the detection of radiological signs."""

    def __init__(
        self,
        model_dir: str,
        conf_threshold: float = 0.01,
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
        self.classes = self.model.CLASSES

        try:
            self.model.test_cfg.rcnn.score_thr = conf_threshold
        except Exception:
            self.model.test_cfg.score_thr = conf_threshold

        # Log the device that is used for the prediction
        if device_ == 'cuda':
            logging.info(f'Device..............: {torch.cuda.get_device_name(0)}')
        else:
            info = get_cpu_info()
            logging.info(f'Device..............: {info["brand_raw"]}')

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

        # Iterate over class detections
        df = pd.DataFrame(columns=columns)
        img_height, img_width = cv2.imread(img_path).shape[:2]
        for class_idx, detections_class in enumerate(detections):
            if detections_class.size == 0:
                num_detections = 1
            else:
                num_detections = detections_class.shape[0]

            # Iterate over boxes on a single image
            df_ = pd.DataFrame(index=range(num_detections), columns=columns)
            df_['Image path'] = img_path
            df_['Image name'] = Path(img_path).name
            df_['Image height'] = img_height
            df_['Image width'] = img_width
            for idx, box in enumerate(detections_class):
                # box -> array(x_min, y_min, x_max, y_max, confidence)
                df_.at[idx, 'x1'] = int(box[0])
                df_.at[idx, 'y1'] = int(box[1])
                df_.at[idx, 'x2'] = int(box[2])
                df_.at[idx, 'y2'] = int(box[3])
                df_.at[idx, 'Feature ID'] = class_idx + 1
                df_.at[idx, 'Feature'] = self.classes[class_idx]
                df_.at[idx, 'Confidence'] = box[4]

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
        model_dir='models/feature_detection/FasterRCNN',
        conf_threshold=0.01,
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
        os.path.join(test_dir, 'predictions2.xlsx'),
        sheet_name='Detections',
        index=True,
        index_label='ID',
    )
