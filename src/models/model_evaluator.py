from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm


class ModelEvaluator:
    """A class used for the detection of radiological signs."""

    def __init__(
        self,
        save_dir: str,
    ) -> None:
        self.save_dir = save_dir

    def combine_data(
        self,
        gt_path: str,
        pred_path: str,
    ) -> Dict[str, Any]:
        # Read the ground truth and the predicted data
        df_gt = pd.read_excel(gt_path)
        df_pred = pd.read_excel(pred_path)

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
        json_path: str,
    ):
        pass


if __name__ == '__main__':
    evaluator = ModelEvaluator(save_dir='data/coco/test_demo')
    b = evaluator.combine_data(
        gt_path='data/coco/test_demo/labels.xlsx',
        pred_path='data/coco/test_demo/predictions.xlsx',
    )
    print(b)
