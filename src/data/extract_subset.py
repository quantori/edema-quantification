import argparse
import logging
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Tuple

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_single_study(
    group: Tuple[int, pd.DataFrame],
    dataset_dir: str,
    save_pairs_only: bool,
    save_dir: str,
) -> pd.DataFrame:
    _, df_study = group
    df_study.reset_index(drop=True, inplace=True)
    subject_id = df_study.iloc[0]['Subject ID']
    study_id = df_study.iloc[0]['Study ID']

    # Process study with paired images
    study_views = [df_study.iloc[idx]['View'] for idx in range(len(df_study))]
    is_correct_view = (
        ('AP' in study_views)
        and ('LATERAL' in study_views)
        or ('PA' in study_views)
        and ('LATERAL' in study_views)
        or ('AP' in study_views)
        and ('LL' in study_views)
        or ('PA' in study_views)
        and ('LL' in study_views)
    )

    # Save paired images of a study
    study_dir = os.path.join(save_dir, 'files', f'{subject_id}', f'{study_id}')
    if save_pairs_only:
        if is_correct_view and len(df_study) == 2:
            os.makedirs(study_dir, exist_ok=True)
            for idx, _src_path in df_study['Image path'].iteritems():
                src_path = os.path.join(dataset_dir, _src_path)
                img_name = Path(src_path).name
                dst_path = os.path.join(study_dir, img_name)
                _dst_path = os.path.relpath(dst_path, start=save_dir)
                df_study.at[idx, 'Image path'] = _dst_path
                df_study.at[idx, 'Image name'] = img_name
                try:
                    shutil.copy(src_path, dst_path)
                except Exception as e:
                    logger.info(f'Error occurred while copying {img_name}')
            return df_study

        else:
            return pd.DataFrame([])

    # Save all images of a study
    else:
        os.makedirs(study_dir, exist_ok=True)
        for idx, _src_path in df_study['Image path'].iteritems():
            src_path = os.path.join(dataset_dir, _src_path)
            img_name = Path(src_path).name
            dst_path = os.path.join(study_dir, img_name)
            _dst_path = os.path.relpath(dst_path, start=save_dir)
            df_study.at[idx, 'Image path'] = _dst_path
            df_study.at[idx, 'Image name'] = img_name
            try:
                shutil.copy(src_path, dst_path)
            except Exception as e:
                logger.info(f'Error occurred while copying {img_name}')
        return df_study


def extract_subset(
    dataset_dir: str,
    finding: str,
    save_pairs_only: bool,
    save_dir: str,
) -> None:
    """Extract a subset with a specified disease from the MIMIC dataset.

    Args:
        dataset_dir: a path to the MIMIC-CXR dataset
        finding: one of the findings to be extracted
        save_pairs_only: if True, save only the paired cases (frontal and lateral images)
        save_dir: directory where the output files will be saved
    Returns:
        None
    """
    logger.info(f'Dataset dir...............: {dataset_dir}')
    logger.info(f'Finding...................: {finding}')
    logger.info(f'Save pairs only...........: {save_pairs_only}')
    logger.info(f'Save dir..................: {save_dir}')

    # Read the source data frame and filter it by the required disease
    metadata_csv = os.path.join(dataset_dir, 'metadata.csv')
    df = pd.read_csv(metadata_csv)
    df_finding = df[df[finding] == 1]

    # Processing of the required dataframe
    groups = df_finding.groupby(['Study ID'])
    processing_func = partial(
        process_single_study,
        dataset_dir=dataset_dir,
        save_pairs_only=save_pairs_only,
        save_dir=save_dir,
    )
    result = Parallel(n_jobs=-1)(
        delayed(processing_func)(group)
        for group in tqdm(groups, desc='Dataset conversion', unit=' study')
    )
    df_out = pd.concat(result)
    df_out.reset_index(drop=True, inplace=True)
    df_out.sort_values(['Subject ID', 'ID'], inplace=True)
    save_path = os.path.join(save_dir, f'metadata.csv')
    df_out.to_csv(
        save_path,
        index=False,
    )
    logger.info(f'Source subjects...........: {len(df_finding["Subject ID"].unique())}')
    logger.info(f'Output subjects...........: {len(df_out["Subject ID"].unique())}')
    logger.info(f'Source studies............: {len(df_finding["Study ID"].unique())}')
    logger.info(f'Output studies............: {len(df_out["Study ID"].unique())}')
    logger.info(f'Source images.............: {len(df_finding["DICOM ID"].unique())}')
    logger.info(f'Output images.............: {len(df_out["DICOM ID"].unique())}')
    logger.info('Complete')


if __name__ == '__main__':
    DISEASES = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Enlarged Cardiomediastinum',
        'Fracture',
        'Lung Lesion',
        'Lung Opacity',
        'No Finding',
        'Pleural Effusion',
        'Pleural Other',
        'Pneumonia',
        'Pneumothorax',
        'Support Devices',
    ]

    parser = argparse.ArgumentParser(description='Extract MIMIC subset')
    parser.add_argument('--dataset_dir', default='data/raw', type=str)
    parser.add_argument('--disease', default='Edema', type=str, choices=DISEASES)
    parser.add_argument('--save_pairs_only', action='store_true')
    parser.add_argument('--save_dir', default='data/edema', type=str)
    args = parser.parse_args()

    extract_subset(
        dataset_dir=args.dataset_dir,
        finding=args.disease,
        save_pairs_only=args.save_pairs_only,
        save_dir=args.save_dir,
    )
