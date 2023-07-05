import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def evaluate_classifiaction(
    df_gt: pd.DataFrame,
    df_pred: pd.DataFrame,
    save_dir: str = './eval/',
    mode: str = 'all',
) -> None:
    """Evaluates classification metrics.

    Saves averaged and class-wise precision, recall, f1, accuracy for the predicted classes of
    pulmonary edema.

    Args:
        df_gt: DataFrame with ground truth labels.
        df_pred: DataFrame with predicted labels.
        save_dir: the directory for saving the output DataFrame.
        mode: defines which data to use {'test', 'train', 'all'}.
    """
    df_gt_preprocessed = _preprocess_df_gt(df_gt, mode)
    df_pred_preprocessed = _preprocess_df_pred(df_pred, mode)
    df_gt_filtered, df_pred_filtered = _filter_dfs(df_gt_preprocessed, df_pred_preprocessed)
    gt_labels, pred_labels = _get_labels(df_gt_filtered, df_pred_filtered)
    df_report = _get_df_report(gt_labels, pred_labels)
    _save_df_report(df_report, save_dir, mode)


def _preprocess_df_gt(df_gt: pd.DataFrame, mode: str) -> pd.DataFrame:
    # Filter the ground truth DataFrame depending on the mode.
    if mode == 'test':
        df_gt.drop_duplicates(subset=['Image name'], keep='first', inplace=True)
        df_gt = df_gt[df_gt['Split'] == 'test']
    elif mode == 'train':
        df_gt.drop_duplicates(subset=['Image name'], keep='first', inplace=True)
        df_gt = df_gt[df_gt['Split'] == 'train']
    else:
        df_gt.drop_duplicates(subset=['Image name'], keep='first', inplace=True)
    return df_gt


def _preprocess_df_pred(df_pred: pd.DataFrame, mode: str) -> pd.DataFrame:
    # Filter the prediction DataFrame and change names in the 'Image name' column.
    df_pred.drop_duplicates(subset=['Image path'], keep='first', inplace=True)
    df_pred['Image name'] = df_pred['Image path'].str.split('\\', expand=True)[1] + '.png'
    return df_pred


def _filter_dfs(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Leave only those rows which are present in both DataFrames.
    df_pred_filtered = df_pred[df_pred['Image name'].isin(df_gt['Image name'])]
    df_gt_filtered = df_gt[df_gt['Image name'].isin(df_pred_filtered['Image name'])]
    return (df_gt_filtered, df_pred_filtered)


def _get_labels(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> Tuple[np.array, np.array]:
    # Returns edema IDs for ground truth and predicition DataFrames.
    gt_labels = df_gt['Class ID'].to_numpy()
    pred_labels = df_pred['Class ID'].to_numpy()
    return (gt_labels, pred_labels)


def _get_df_report(gt_labels: np.array, pred_labels: np.array) -> pd.DataFrame:
    # Returns DataFrame with main metrics (precision, recall, f1, accuracy). Based on
    # classification_report() function of scikit-learn.
    target_names = ('No edema', 'Vascular congestion', 'Interstitial edema', 'Alveolar edema')
    report = classification_report(
        gt_labels,
        pred_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def _save_df_report(df: pd.DataFrame, save_dir: str, mode: str) -> None:
    metrics_path = os.path.join(save_dir, f'classification_metrics_{mode}.xlsx')
    df.to_excel(metrics_path, sheet_name='Classification metrics')


if __name__ == '__main__':
    df_gt = pd.read_excel('./data/interim/metadata.xlsx')
    df_pred = pd.read_excel('./data/interim_predict/metadata.xlsx')
    evaluate_classifiaction(df_gt, df_pred, mode='train')
