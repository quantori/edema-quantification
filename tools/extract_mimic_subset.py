import os
import logging
import argparse
import multiprocessing
from pathlib import Path
from functools import partial

import pandas as pd
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
        group: str,
        save_dir: str,
) -> pd.DataFrame:
    _, df_study = group

    # TODO: process study based on the number of images
    if len(df_study) > 1:
        print('One image')
    else:
        print('One image')

    return pd.DataFrame([1])


def extract_subset(
        metadata_csv: str,
        disease: str,
        view: str,
        save_dir: str,
) -> None:

    """

    Args:
        metadata_csv: a path to a MIMIC metadata CSV file
        disease: one of the diseases to be extracted
        view: one of the image views PA (postero-anterior), LATERAL (lateral), AP (antero-posterior)
        save_dir: directory where the output files will be saved

    Returns:
        None
    """

    assert disease in DISEASES, f'Incorrect disease: {disease}. Should be one of {DISEASES}'

    logger.info(f'Metadata CSV..............: {metadata_csv}')
    logger.info(f'Disease...................: {disease}')
    logger.info(f'View......................: {view}')
    logger.info(f'Save dir..................: {save_dir}')

    # Read the source data frame and extract the required subset
    df = pd.read_csv(metadata_csv)
    if view in VIEWS:
        df_out = df[df[disease] == 1 & df['View'] == view]
    else:
        df_out = df[df[disease] == 1]
    logger.info(f'Number of images..........: {len(df_out["DICOM ID"].unique())}')
    logger.info(f'Number of subjects........: {len(df_out["Subject ID"].unique())}')
    logger.info(f'Number of studies.........: {len(df_out["Study ID"].unique())}')

    # TODO: Processing of the required dataframe
    groups = df_out.groupby(['Study ID'])
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    processing_func = partial(
        process_single_study,
        save_dir=save_dir,
    )
    a = pool.map(processing_func, tqdm(groups, desc='Dataset conversion', unit=' study'))


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

    VIEWS = [
        'PA',
        'LATERAL',
        'AP',
    ]

    parser = argparse.ArgumentParser(description='Extract MIMIC subset')
    parser.add_argument('--metadata_csv', default='dataset/mimic/metadata_chexpert.csv', type=str)
    parser.add_argument('--disease', default='Edema', type=str, choices=DISEASES)
    parser.add_argument('--view', default=None, type=str, choices=VIEWS)
    parser.add_argument('--save_dir', default='dataset/mimic_edema', type=str)
    args = parser.parse_args()

    extract_subset(
        metadata_csv=args.metadata_csv,
        disease=args.disease,
        view=args.view,
        save_dir=args.save_dir,
    )
