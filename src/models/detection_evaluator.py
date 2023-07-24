from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import fiftyone as fo
import pandas as pd


class DetectionEvaluator:
    """A class used for the detection of radiological signs."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.01,
    ) -> None:
        assert 0 <= iou_threshold <= 1, 'iou_threshold must lie within [0, 1]'
        assert 0 <= conf_threshold <= 1, 'conf_threshold must lie within [0, 1]'
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def combine_data(
        self,
        gt_path: str,
        pred_path: str,
        exclude_features: List[str],
    ) -> Dict[str, Any]:
        # Read ground truth and predictions
        df_gt = pd.read_excel(gt_path)
        df_pred = pd.read_excel(pred_path)
        df_pred = df_pred[df_pred['Confidence'] >= self.conf_threshold]
        df_pred = DetectionEvaluator._preprocess_df_pred(df_pred)
        if len(exclude_features) > 0:
            df_gt = df_gt[~df_gt['Feature'].isin(exclude_features)]
            # df_pred = df_pred[~df_pred['Feature'].isin(exclude_features)]
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
                lung_mask='fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Segmentation)',
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

            dets_gt = self._process_detections(df=df_gt_sample)
            dets_pred = self._process_detections(df=df_pred_sample)

            mask_path = img_path.replace('img_crop', 'mask_crop')
            mask_crop = cv2.imread(mask_path)

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
                    lung_mask=dict(
                        _cls='Segmentation',
                        mask_path=mask_path,
                        mask=mask_crop,
                    ),
                ),
            )

        dataset.update(dict(samples=samples))  # type: ignore

        return dataset

    @staticmethod
    def _preprocess_df_pred(
        df_pred: pd.DataFrame,
    ) -> pd.DataFrame:
        # Change names in the 'Image name' column.
        df_pred['Image name'] = df_pred.apply(
            func=lambda row: f'{Path(str(row["Image path"])).parts[-2]}.png',
            axis=1,
        )
        return df_pred

    @staticmethod
    def _process_detections(
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
                area=row['Box area'],
            )

            if 'Confidence' in df.columns:
                det.update(dict(confidence=row['Confidence']))

            detections.append(det)
        return detections

    def evaluate(
        self,
        detections: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Create dataset
        dataset = fo.Dataset(
            name=None,
            persistent=False,
            overwrite=True,
        )
        dataset = dataset.from_dict(detections)

        results = dataset.evaluate_detections(
            pred_field='predictions',
            gt_field='ground_truth',
            eval_key='eval',
            iou=self.iou_threshold,
        )

        dataset.count_values('ground_truth.detections.label')

        # Confusion matrix (use for optimal threshold values)
        # cm = results.confusion_matrix(classes=classes)
        # plot = results.plot_confusion_matrix()
        # plot.show()

        # Сalculate metrics
        df_metrics = self._calculate_metrics(results)
        df_metrics_cw = self._calculate_metrics_class_wise(results)

        return df_metrics, df_metrics_cw

    def _calculate_metrics(
        self,
        results: fo.DetectionResults,
    ) -> pd.DataFrame:
        metrics = results.metrics(classes=None)
        df = pd.DataFrame([metrics])
        df['TP'] = results._samples.sum('eval_tp')
        df['FP'] = results._samples.sum('eval_fp')
        df['FN'] = results._samples.sum('eval_fn')
        df['IoU'] = self.iou_threshold
        df['Confidence'] = self.conf_threshold
        df.rename(
            columns={
                'accuracy': 'Accuracy',
                'precision': 'Precision',
                'recall': 'Recall',
                'fscore': 'F1',
                'support': 'Support',
            },
            inplace=True,
        )

        return df

    def _calculate_metrics_class_wise(
        self,
        results: fo.DetectionResults,
    ) -> pd.DataFrame:
        metrics = results.report(classes=None)
        df = pd.DataFrame.from_dict(metrics, orient='index')
        df = df.drop(['micro avg', 'macro avg', 'weighted avg'])
        df['IoU'] = self.iou_threshold
        df['Confidence'] = self.conf_threshold
        df.reset_index(inplace=True)
        df.rename(
            columns={
                'index': 'Feature',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1-score': 'F1',
                'support': 'Support',
            },
            inplace=True,
        )

        return df

    def visualize(
        self,
        detections: Dict[str, Any],
    ) -> None:
        # Create dataset
        dataset = fo.Dataset(
            name=None,
            persistent=False,
            overwrite=True,
        )
        dataset = dataset.from_dict(detections)

        # Visualize dataset
        session = fo.launch_app(dataset=dataset)
        session.wait()
        dataset.delete()