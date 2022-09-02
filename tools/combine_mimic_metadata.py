import os
import logging
import argparse
from pathlib import Path

import pandas as pd
from utils import get_file_list

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def combine_metadata(
        metadata_csv: str,
        findings_csv: str,
        split_csv: str,
        save_dir: str,
        dataset_dir: str = None,
) -> None:

    df_findings = pd.read_csv(findings_csv, compression='gzip')
    df_metadata = pd.read_csv(metadata_csv, compression='gzip')
    df_split = pd.read_csv(split_csv, compression='gzip')
    os.makedirs(save_dir, exist_ok=True)

    _df_out = df_metadata.merge(df_findings, on=['subject_id', 'study_id'], how='left')
    df_out = _df_out.join(df_split['split'])

    logger.info(f'Metadata CSV..............: {metadata_csv}')
    logger.info(f'Findings CSV..............: {findings_csv}')
    logger.info(f'Split CSV.................: {split_csv}')
    logger.info(f'Dataset dir...............: {dataset_dir}')
    logger.info(f'Save dir..................: {save_dir}')

    if dataset_dir:
        img_paths = get_file_list(
            src_dirs=dataset_dir,
            ext_list=[
                '.png',
                '.jpg',
                '.jpeg',
                '.bmp',
            ]
        )
        num_images = len(img_paths)

        dicom_ids = [str(Path(img_path).stem) for img_path in img_paths]
        df_paths = pd.DataFrame(list(zip(dicom_ids, img_paths)), columns=['dicom_id', 'Image path'])
        df_out = df_out.merge(df_paths, on=['dicom_id'], how='left')
        df_out['Image name'] = df_out.apply(lambda row: Path(str(row['Image path'])).name, axis=1)
    else:
        num_images = None

    logger.info(f'Number of images..........: {num_images}')

    suffix = 'chexpert' if 'chexpert' in findings_csv else 'negbio'
    meta_path = os.path.join(save_dir, f'metadata_{suffix}.csv')
    cols = {
        'dicom_id': 'DICOM ID',
        'subject_id': 'Subject ID',
        'study_id': 'Study ID',
        'PerformedProcedureStepDescription': 'Description',
        'ViewPosition': 'View',
        'Rows': 'Height',
        'Columns': 'Width',
        'StudyDate': 'Study Date',
        'StudyTime': 'Study Time',
        'ProcedureCodeSequence_CodeMeaning': 'Procedure Code',
        'ViewCodeSequence_CodeMeaning': 'View Code',
        'PatientOrientationCodeSequence_CodeMeaning': 'Patient Orientation',
        'split': 'Split',
    }
    df_out.rename(columns=cols, inplace=True)
    df_out.index += 1
    df_out.to_csv(
        meta_path,
        index=True,
        index_label='ID',
        encoding='utf-8',
    )
    logger.info('Complete')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combine metadata')
    parser.add_argument('--metadata_csv', default='dataset/mimic/mimic-cxr-2.0.0-metadata.csv.gz', type=str)
    parser.add_argument('--findings_path', default='dataset/mimic/mimic-cxr-2.0.0-chexpert.csv.gz', type=str)
    parser.add_argument('--split_csv', default='dataset/mimic/mimic-cxr-2.0.0-split.csv.gz', type=str)
    parser.add_argument('--dataset_dir', default=None, type=str)
    parser.add_argument('--save_dir', default='dataset/mimic', type=str)
    args = parser.parse_args()

    combine_metadata(
        metadata_csv=args.metadata_csv,
        findings_csv=args.findings_path,
        split_csv=args.split_csv,
        save_dir=args.save_dir,
        dataset_dir=args.dataset_dir,
    )
