import logging
import os.path as osp
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from cpuinfo import get_cpu_info
from mmdet.apis import inference_detector, init_detector

from src.data.utils import get_file_list


class SignDetector:
    """A class used for the detection of radiological signs."""

    def __init__(
        self,
        model_dir: str,
        conf_threshold: float = 0.01,
        device: str = 'auto',
    ):
        # Get config path
        # TODO: two options available:
        #       + (1) specify a path to a directory with the config and a checkpoint in it
        #       (2) specify both a path to a config and a path to a checkpoint
        config_list = get_file_list(
            src_dirs=model_dir,
            ext_list='.py',
        )
        assert len(config_list) == 1, 'Keep only one config file in the model directory'
        config_path = config_list[0]
        self.config_name, _ = osp.splitext(osp.basename(config_path))

        # Get checkpoint path
        # TODO: I think there will be several checkpoints
        #       If the path is not specified, use a default checkpoint
        checkpoint_list = get_file_list(
            src_dirs=model_dir,
            ext_list='.pth',
        )
        if len(checkpoint_list) > 1:
            checkpoint_list = [cp for cp in checkpoint_list if 'latest' in cp]
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
        self.model.test_cfg.rcnn.score_thr = conf_threshold

        # Log the device that is used for the prediction
        if device_ == 'cuda':
            logging.info(f'Device..............: {torch.cuda.get_device_name(0)}')
        else:
            info = get_cpu_info()
            logging.info(f'Device..............: {info["brand_raw"]}')

    # TODO: test if this method works properly
    # changed predict -> call
    def __call__(
        self,
        img_paths: List[str],
    ) -> List[List[np.ndarray]]:
        detections = inference_detector(
            model=self.model,
            imgs=img_paths,
        )

        return detections

    # TODO: test if this method works properly
    def process_detections(
        self,
        img_paths: List[str],
        detections: List[List[np.ndarray]],
    ) -> pd.DataFrame:
        columns = [
            'img_path',
            'img_name',
            'img_height',
            'img_width',
            'x1',
            'y1',
            'x2',
            'y2',
            'class_id',
            'class',
            'confidence',
        ]

        # Iterate over images
        df = pd.DataFrame(columns=columns)
        for image_idx, (img_path, detections_image) in enumerate(zip(img_paths, detections)):
            # Iterate over class detections
            for class_idx, detections_class in enumerate(detections_image):
                if detections_class.size == 0:
                    num_detections = 1
                else:
                    num_detections = detections_class.shape[0]

                # Iterate over boxes on a single image
                df_ = pd.DataFrame(index=range(num_detections), columns=columns)
                df_['img_path'] = img_path
                df_['img_name'] = Path(img_path).name
                for idx, box in enumerate(detections_class):
                    # box -> array(x_min, y_min, x_max, y_max, confidence)
                    df_.at[idx, 'x1'] = int(box[0])
                    df_.at[idx, 'y1'] = int(box[1])
                    df_.at[idx, 'x2'] = int(box[2])
                    df_.at[idx, 'y2'] = int(box[3])
                    df_.at[idx, 'class_id'] = class_idx
                    df_.at[idx, 'class'] = self.classes[class_idx]
                    df_.at[idx, 'confidence'] = box[4]
                df = pd.concat([df, df_])

        df.sort_values('img_path', inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df


if __name__ == '__main__':
    img_paths = ['data/demo/input/10000032_50414267.png']
    model = SignDetector(
        model_dir=f'models/sign_detection/FasterRCNN_213957_260423',
        conf_threshold=0.01,
        device='auto',
    )
    mask = model(img_paths)
    res_det = model.process_detections(img_paths=img_paths, detections=mask)

    res_det.to_csv(f'data/sigh_detector/mask_{model.config_name}.csv')
