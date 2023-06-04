import json
import logging
import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def detections_by_img_name(img_name: str, df: pd.DataFrame) -> []:
    detections = []
    df_filtered = df[df['Image name'] == img_name]
    for index, row in df_filtered.iterrows():
        if row['Feature'] is not np.NAN:
            det = dict(
                _cls='Detection',
                tags=[],
                attributes=dict(),
                label=row['Feature'],
                bounding_box=[
                    row['x1'] / row['Image width'],
                    row['y1'] / row['Image height'],
                    row['Box width'] / row['Image width'],
                    row['Box height'] / row['Image height'],
                ],
                area=row['Box area'],
            )

            if 'Confidence' in df_filtered.columns:
                det.update(dict(confidence=row['Confidence']))
            detections.append(det)
    return detections


def process_data(
    gt_path: str,
    pred_path: str,
):
    use_columns = [
        'Image name',
        'Image width',
        'Image height',
        'Feature',
        'Box area',
        'Box width',
        'Box height',
        'x1',
        'y1',
        'x2',
        'y2',
    ]
    ground_truth = pd.read_excel(gt_path, usecols=use_columns)

    predictions = pd.read_excel(pred_path, usecols=use_columns + ['Confidence'])
    dataset = dict(
        name='edema',
        media_type='image',
        sample_fields=dict(
            filepath='fiftyone.core.fields.StringField',
            metadata='fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.ImageMetadata)',
            ground_truth='fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)',
            predictions='fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)',
        ),
        info=dict(),
    )
    samples = []
    predictions_unique_image_names = predictions['Image name'].unique()
    for img_name in predictions_unique_image_names:
        samples.append(
            dict(
                filepath=os.path.join('data', 'coco', 'test', 'data', img_name),
                tags=['validation'],
                metadata=None,
                ground_truth=dict(
                    _cls='Detections',
                    detections=detections_by_img_name(
                        img_name=img_name,
                        df=ground_truth,
                    ),
                ),
                predictions=dict(
                    _cls='Detections',
                    detections=detections_by_img_name(
                        img_name=img_name,
                        df=predictions,
                    ),
                ),
            ),
        )

    dataset.update(dict(samples=samples))

    return dataset


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='combine_gt_pred',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    dataset = process_data(
        gt_path=cfg.gt_path,
        pred_path=cfg.pred_path,
    )

    with open(cfg.save_path, 'w') as file:
        json.dump(dataset, file)

    log.info('Complete')


if __name__ == '__main__':
    main()
