import logging
import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from utils import get_file_list

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='combine_metadata',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    df_findings = pd.read_csv(cfg.findings_csv, compression='gzip')
    df_metadata = pd.read_csv(cfg.metadata_csv, compression='gzip')
    df_split = pd.read_csv(cfg.split_csv, compression='gzip')
    os.makedirs(cfg.save_dir, exist_ok=True)

    _df_out = df_metadata.merge(df_findings, on=['subject_id', 'study_id'], how='left')
    df_out = _df_out.join(df_split['split'])

    log.info(f'Metadata CSV..............: {cfg.metadata_csv}')
    log.info(f'Findings CSV..............: {cfg.findings_csv}')
    log.info(f'Split CSV.................: {cfg.split_csv}')
    log.info(f'Dataset dir...............: {cfg.dataset_dir}')
    log.info(f'Save dir..................: {cfg.save_dir}')

    if cfg.dataset_dir:
        img_paths = get_file_list(
            src_dirs=cfg.dataset_dir,
            ext_list=[
                '.png',
                '.jpg',
                '.jpeg',
                '.bmp',
            ],
        )
        num_images = len(img_paths)

        dicom_ids = [str(Path(img_path).stem) for img_path in img_paths]
        df_paths = pd.DataFrame(list(zip(dicom_ids, img_paths)), columns=['dicom_id', 'Image path'])
        df_paths['Image path'] = df_paths.apply(
            func=lambda row: os.path.relpath(str(row['Image path']), start=cfg.save_dir),
            axis=1,
        )
        df_paths['Image name'] = df_paths.apply(
            func=lambda row: Path(str(row['Image path'])).name,
            axis=1,
        )
        df_out = df_out.merge(df_paths, on=['dicom_id'], how='left')
    else:
        num_images = None

    log.info(f'Number of images..........: {num_images}')

    meta_path = os.path.join(cfg.save_dir, 'metadata.csv')
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
    log.info('Complete')


if __name__ == '__main__':
    main()
