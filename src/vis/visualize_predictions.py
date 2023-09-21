import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import fiftyone as fo
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def combine_data(
    gt_path: str,
    pred_path: str,
    conf_threshold: float,
    exclude_features: List[str],
) -> Dict[str, Any]:
    # Read ground truth and predictions
    df_gt = pd.read_excel(gt_path)
    df_pred = pd.read_excel(pred_path)
    df_pred = df_pred[df_pred['Confidence'] >= conf_threshold]
    df_pred['Image name'] = df_pred.apply(
        func=lambda row: f'{Path(str(row["Image path"])).parts[-2]}.png',
        axis=1,
    )
    if len(exclude_features) > 0:
        df_gt = df_gt[~df_gt['Feature'].isin(exclude_features)]
        df_pred = df_pred[~df_pred['Feature'].isin(exclude_features)]
        df_pred = df_pred[df_pred['Image name'].isin(df_gt['Image name'])]

    # Initialization of the fiftyone dataset
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

    # Process boxes one image at a time
    samples = []
    img_paths = df_pred['Image path'].unique()
    for img_path in img_paths:
        img_id = Path(img_path).parts[-2]
        df_gt_sample = df_gt[df_gt['Image path'].str.contains(img_id)]
        df_pred_sample = df_pred[df_pred['Image path'].str.contains(img_id)]

        dets_gt = process_detections(df=df_gt_sample)
        dets_pred = process_detections(df=df_pred_sample)

        split = df_gt_sample['Split'].unique()[0]
        samples.append(
            dict(
                filepath=img_path,
                tags=[split],
                metadata=None,
                ground_truth=dict(
                    _cls='Detections',
                    detections=dets_gt,
                ),
                predictions=dict(
                    _cls='Detections',
                    detections=dets_pred,
                ),
            ),
        )

    dataset.update(dict(samples=samples))  # type: ignore

    return dataset


def process_detections(
    df: pd.DataFrame,
) -> List:
    detections = []
    for idx, row in df.iterrows():
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
        )

        if 'Confidence' in df.columns:
            det.update(dict(confidence=row['Confidence']))

        detections.append(det)
    return detections


def visualize(
    detections: Dict[str, Any],
    save_dir: str,
    save_images: bool = False,
) -> None:
    # Create dataset
    dataset = fo.Dataset(
        name=None,
        persistent=False,
        overwrite=True,
    )
    dataset = dataset.from_dict(detections)

    # Save images
    if save_images:
        os.makedirs(save_dir, exist_ok=True)
        dataset.draw_labels(
            save_dir,
            label_fields=['ground_truth', 'predictions'],
            overwrite=True,
            show_object_labels=True,
            show_all_confidences=True,
            per_object_label_colors=True,
            bbox_alpha=0.75,
            bbox_linewidth=3,
        )

    # Visualize dataset
    session = fo.launch_app(dataset=dataset)
    session.wait()
    dataset.delete()


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='visualize_predictions',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Check conf_threshold
    assert 0 <= cfg.conf_threshold <= 1, 'conf_threshold must lie within [0, 1]'

    # Combine ground truth and predictions
    dets = combine_data(
        gt_path=cfg.gt_path,
        pred_path=cfg.pred_path,
        conf_threshold=cfg.conf_threshold,
        exclude_features=cfg.exclude_features,
    )

    # Visually compare ground truth and predictions
    visualize(
        detections=dets,
        save_images=cfg.save_images,
        save_dir=cfg.save_dir,
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
