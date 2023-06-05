from pathlib import Path
from typing import Any, Dict, List, Tuple

import fiftyone as fo
import pandas as pd
from tqdm import tqdm


class ModelEvaluator:
    """A class used for the detection of radiological signs."""

    def __init__(
        self,
        iou_thresh: float = 0.5,
        confidence_thresh: float = 0.01,
    ) -> None:
        assert 0 <= iou_thresh <= 1, 'iou_thresh must lie within [0, 1]'
        assert 0 <= confidence_thresh <= 1, 'confidence_thresh must lie within [0, 1]'
        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh

    def combine_data(
        self,
        gt_path: str,
        pred_path: str,
        exclude_features: List[str] = None,
    ) -> Dict[str, Any]:
        # Read the ground truth and the predicted data
        df_gt = pd.read_excel(gt_path)
        df_pred = pd.read_excel(pred_path)
        df_pred = df_pred[df_pred['Confidence'] >= self.confidence_thresh]
        if exclude_features is not None:
            df_gt = df_gt[~df_gt['Feature'].isin(exclude_features)]
            df_pred = df_pred[~df_pred['Feature'].isin(exclude_features)]

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
        for img_path in tqdm(img_paths, desc='Processing data', unit=' images'):
            img_name = Path(img_path).name
            df_gt_sample = df_gt[df_gt['Image name'] == img_name]
            df_pred_sample = df_pred[df_pred['Image name'] == img_name]

            dets_gt = self._process_detections(df=df_gt_sample)
            dets_pred = self._process_detections(df=df_pred_sample)

            samples.append(
                dict(
                    filepath=img_path,
                    tags=['validation'],
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

        dataset.update(dict(samples=samples))

        return dataset

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
        top_class_count: int = 7,
    ) -> Tuple[int, int, int]:
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
            iou=self.iou_thresh,
        )

        counts = dataset.count_values('ground_truth.detections.label')
        classes = sorted(counts, key=counts.get, reverse=True)[:top_class_count]
        results.print_report(classes=classes)

        tp = dataset.sum('eval_tp')
        fp = dataset.sum('eval_fp')
        fn = dataset.sum('eval_fn')
        print(f'TP: {tp}')
        print(f'FP: {fp}')
        print(f'FN: {fn}')

        return tp, fp, fn

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


if __name__ == '__main__':
    evaluator = ModelEvaluator(
        iou_thresh=0.5,
        confidence_thresh=0.5,
    )
    dets = evaluator.combine_data(
        gt_path='data/coco/test_demo/labels.xlsx',
        pred_path='data/coco/test_demo/predictions.xlsx',
        exclude_features=None,
    )
    tp, fp, fn = evaluator.evaluate(detections=dets)
    evaluator.visualize(detections=dets)
    print('Complete')
