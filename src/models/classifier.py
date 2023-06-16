from typing import List, Tuple

import numpy as np
import pandas as pd

from src.data.utils_sly import CLASS_MAP


class EdemaClassifier:
    """Hard-coded edema classifier."""

    def __init__(self) -> None:
        self._output: List = []

    def classify(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """The main classification function.

        Args:
            df: initial dataframe containing images' IDs and features.

        Returns:
            DataFrame with identified edema severity class.
        """
        image_names = self._get_unique_image_names(df_in)
        for idx, image_name in enumerate(image_names):
            df_temp = df_in[df_in['Image name'] == image_name]
            edema_severity, feature, confidence = EdemaClassifier._get_edema_severity(df_temp)
            new_row = {
                'ID': idx,
                'Image path': df_temp['Image path'].iloc[0],
                'Image name': image_name,
                'Image height': df_temp['Image height'].iloc[0],
                'Image width': df_temp['Image width'].iloc[0],
                'Feature': feature,
                'Confidence': confidence,
                'Class ID': CLASS_MAP[edema_severity],
                'Class': edema_severity,
            }
            self._output.append(new_row)
        return pd.DataFrame(self._output)

    def _get_unique_image_names(self, df: pd.DataFrame) -> np.ndarray:
        return pd.unique(df['Image name'])

    @staticmethod
    def _get_edema_severity(df: pd.DataFrame) -> Tuple[str, float, float]:
        features = df['Feature'].unique()
        if 'Bat' in features or 'Infiltrate' in features:
            edema_severity, feature, confidence = EdemaClassifier._get_alveolar_edema(df)
            return edema_severity, feature, confidence

        elif 'Effusion' in features or 'Kerley' in features or 'Cuffing' in features:
            edema_severity, feature, confidence = EdemaClassifier._get_interstitial_edema(df)
            return edema_severity, feature, confidence

        elif 'Cephalization' in features:
            edema_severity, feature, confidence = EdemaClassifier._get_vascular_congestion(df)
            return edema_severity, feature, confidence

        else:
            return 'No edema', np.nan, np.nan

    @staticmethod
    def _get_alveolar_edema(df: pd.DataFrame) -> Tuple[str, float, float]:
        df_alv_edema = df[(df['Feature'] == 'Bat') | (df['Feature'] == 'Infiltrate')]
        max_row = df_alv_edema.nlargest(1, ['Confidence'])
        return 'Alveolar edema', max_row['Feature'].values[0], max_row['Confidence'].values[0]

    @staticmethod
    def _get_interstitial_edema(df: pd.DataFrame) -> Tuple[str, float, float]:
        df_int_edema = df[
            (df['Feature'] == 'Effusion')
            | (df['Feature'] == 'Kerley')
            | (df['Feature'] == 'Cuffing')
        ]
        max_row = df_int_edema.nlargest(1, ['Confidence'])
        return (
            'Interstitial edema',
            max_row['Feature'].values[0],
            max_row['Confidence'].values[0],
        )

    @staticmethod
    def _get_vascular_congestion(df: pd.DataFrame) -> Tuple[str, float, float]:
        df_cong_edema = df[df['Feature'] == 'Cephalization']
        max_row = df_cong_edema.nlargest(1, ['Confidence'])
        return (
            'Vascular congestion',
            max_row['Feature'].values[0],
            max_row['Confidence'].values[0],
        )


if __name__ == '__main__':
    df = pd.read_excel('./data/coco/test/predictions.xlsx')
    classifier = EdemaClassifier()
    df_o = classifier.classify(df)
    print(df_o)
