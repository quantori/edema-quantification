import logging
import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import get_file_list
from src.models.box_fuser import BoxFuser
from src.models.edema_classifier import EdemaClassifier
from src.models.edema_net import EdemaNet
from src.models.feature_detector import FeatureDetector
from src.models.lung_segmenter import LungSegmenter
from src.models.map_fuser import MapFuser
from src.models.mask_processor import MaskProcessor
from src.models.non_max_suppressor import NonMaxSuppressor

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='predict',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Get list of images to predict
    img_paths = get_file_list(
        src_dirs=cfg.data_dir,
        ext_list=[
            '.png',
            '.jpg',
            '.jpeg',
            '.bmp',
        ],
    )
    log.info(f'Number of images..........: {len(img_paths)}')

    # Initialize lung segmentation models
    lung_segmenters = []
    for model_dir in cfg.seg_model_dirs:
        lung_segmenters.append(
            LungSegmenter(
                model_dir=model_dir,
                device='auto',
            ),
        )

    # Initialize feature detection models
    feature_detectors = []
    for model_dir in cfg.det_model_dirs:
        feature_detectors.append(
            FeatureDetector(
                model_dir=model_dir,
                conf_threshold=0.01,
                device='auto',
            ),
        )

    # Initialize probability map fuser
    map_fuser = MapFuser()

    # Initialize binary mask processor
    mask_processor = MaskProcessor(
        threshold_method='otsu',
        kernel_size=(7, 7),
    )

    # Initialize non-maximum suppressor
    non_max_suppressor = NonMaxSuppressor(
        method=cfg.nms_method,
        sigma=0.1,
        iou_threshold=cfg.iou_threshold,
        conf_thresholds=cfg.conf_thresholds,
    )

    # Initialize box fuser
    box_fuser = BoxFuser(
        iou_threshold=cfg.iou_threshold,
    )

    # Initialize edema classifier
    edema_classifier = EdemaClassifier()

    edema_net = EdemaNet(
        lung_segmenters=lung_segmenters,
        feature_detectors=feature_detectors,
        map_fuser=map_fuser,
        mask_processor=mask_processor,
        non_max_suppressor=non_max_suppressor,
        box_fuser=box_fuser,
        edema_classifier=edema_classifier,
        img_size=cfg.img_size,
        lung_extension=cfg.lung_extension,
    )

    df = pd.DataFrame()
    for img_path in tqdm(img_paths, desc='Prediction', unit=' images'):
        log.info(f'Processing: {Path(img_path).stem}')
        df_img = edema_net.predict(
            img_path=img_path,
            save_dir=cfg.save_dir,
        )
        df = pd.concat([df, df_img])

    # Save metadata
    metadata_path = os.path.join(cfg.save_dir, 'metadata.xlsx')
    df.index += 1
    df.to_excel(
        metadata_path,
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
