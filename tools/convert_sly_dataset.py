import os
import shutil
import logging
import argparse
from pathlib import Path
from typing import Tuple
from functools import partial
from joblib import Parallel, delayed

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


def main(
        dataset_dir: str,
        save_dir: str,
) -> None:
    print('Saving dataset')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert Supervisely dataset')
    parser.add_argument('--dataset_dir', default='dataset/MIMIC-CXR-Edema-SLY', type=str)
    parser.add_argument('--save_dir', default='dataset/MIMIC-CXR-Edema-Convert', type=str)
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        save_dir=args.save_dir,
    )
