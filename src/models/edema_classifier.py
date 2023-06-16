from typing import List

import pandas as pd

from src.data.utils_sly import CLASS_MAP


class EdemaClassifier:
    """A classifier that assigns an edema class to an X-ray image."""

    def __init__(self) -> None:
        self._output: List = []

    def classify(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """The main classification function.

        Args:
            df: initial dataframe containing image and feature metadata
        Returns:
            df_out: a dataframe with identified edema severity class
        """
        if df.empty:
            raise Exception('DataFrame is empty!')

        img_groups = df.groupby('Image name')
        for _, df_img in img_groups:
            edema_severity = EdemaClassifier._get_edema_severity(df_img)
            df_img['Class ID'] = CLASS_MAP[edema_severity]
            df_img['Class'] = edema_severity
            self._output.append(df_img)
        df_out = pd.concat(self._output)
        return df_out

    @staticmethod
    def _get_edema_severity(df: pd.DataFrame) -> str:
        features = list(df['Feature'].unique())
        if 'Bat' in features or 'Infiltrate' in features:
            return 'Alveolar edema'

        elif 'Effusion' in features or 'Kerley' in features or 'Cuffing' in features:
            return 'Interstitial edema'

        elif 'Cephalization' in features:
            return 'Vascular congestion'

        else:
            return 'No edema'


if __name__ == '__main__':
    df = pd.read_excel('./data/coco/test/predictions.xlsx')
    # df = pd.DataFrame(columns=METADATA_COLUMNS)
    print(df)
    classifier = EdemaClassifier()
    df_o = classifier.classify(df)
    print(df_o)
