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


def get_conf_thresholds(
    model_name: str,
) -> dict:
    if model_name == 'ATSS':
        conf_thresholds = {
            'Cephalization': 0.27,
            'Kerley': 0.54,
            'Effusion': 0.36,
            'Bat': 0.21,
            'Infiltrate': 0.48,
        }
    elif model_name == 'Cascade_RPN':
        conf_thresholds = {
            'Cephalization': 0.42,
            'Kerley': 0.70,
            'Effusion': 0.96,
            'Bat': 0.50,
            'Infiltrate': 0.75,
        }
    elif model_name == 'Faster_RCNN':
        conf_thresholds = {
            'Cephalization': 0.71,
            'Kerley': 0.65,
            'Effusion': 0.77,
            'Bat': 0.62,
            'Infiltrate': 0.99,
        }
    elif model_name == 'FSAF':
        conf_thresholds = {
            'Cephalization': 0.44,
            'Kerley': 0.54,
            'Effusion': 0.47,
            'Bat': 0.45,
            'Infiltrate': 0.39,
        }
    elif model_name == 'GFL':
        conf_thresholds = {
            'Cephalization': 0.32,
            'Kerley': 0.60,
            'Effusion': 0.51,
            'Bat': 0.35,
            'Infiltrate': 0.72,
        }
    elif model_name == 'PAA':
        conf_thresholds = {
            'Cephalization': 0.46,
            'Kerley': 0.48,
            'Effusion': 0.71,
            'Bat': 0.56,
            'Infiltrate': 0.71,
        }
    elif model_name == 'SABL':
        conf_thresholds = {
            'Cephalization': 0.37,
            'Kerley': 0.53,
            'Effusion': 0.77,
            'Bat': 0.48,
            'Infiltrate': 0.38,
        }
    elif model_name == 'TOOD':
        conf_thresholds = {
            'Cephalization': 0.25,
            'Kerley': 0.41,
            'Effusion': 0.36,
            'Bat': 0.32,
            'Infiltrate': 0.24,
        }
    else:
        raise ValueError(f'Unknown model: {model_name}')

    return conf_thresholds


def combine_data(
    gt_path: str,
    pred_path: str,
    conf_thresholds: dict,
    include_features: List[str],
) -> Dict[str, Any]:
    # Read ground truth and predictions
    df_gt = pd.read_excel(gt_path)
    df_pred = pd.read_excel(pred_path)
    df_pred['Image name'] = df_pred.apply(
        func=lambda row: f'{Path(str(row["Image path"])).parts[-2]}.png',
        axis=1,
    )

    # Include only specified features
    if len(include_features) > 0:
        df_gt = df_gt[df_gt['Feature'].isin(include_features)]
        df_pred = df_pred[df_pred['Feature'].isin(include_features)]
        df_pred = df_pred[df_pred['Image name'].isin(df_gt['Image name'])]

    # Filter boxes by feature confidence
    df_pred = filter_dataframe(df_pred, conf_thresholds)

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


def filter_dataframe(
    df: pd.DataFrame,
    conf_thresholds: dict,
) -> pd.DataFrame:
    df_filtered = pd.DataFrame()
    for key, threshold in conf_thresholds.items():
        mask = (df['Feature'] == key) & (df['Confidence'] >= threshold)
        df_filtered = df_filtered.append(df[mask])
    return df_filtered


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
        # Config: https://docs.voxel51.com/api/fiftyone.utils.annotations.html#fiftyone.utils.annotations.DrawConfig
        dataset.draw_labels(
            save_dir,
            label_fields=['ground_truth', 'predictions'],
            overwrite=True,
            show_object_names=False,
            show_object_labels=False,
            show_object_confidences=True,
            per_object_name_colors=True,
            per_object_label_colors=False,
            per_object_index_colors=True,
            bbox_alpha=1.0,
            bbox_linewidth=2,
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

    # Get optimal thresholds
    model_name = Path(cfg.pred_path).parent.name
    conf_thresholds = get_conf_thresholds(model_name)

    # Combine ground truth and predictions
    dets = combine_data(
        gt_path=cfg.gt_path,
        pred_path=cfg.pred_path,
        conf_thresholds=conf_thresholds,
        include_features=cfg.include_features,
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
