import pandas as pd
import os
import argparse

from utils import get_file_list


def combine_metadata(
        metadata_csv: str,
        findings_path: str,
        split_csv: str,
        output_dir: str,
        dataset_dir: str = None,
) -> None:
    output_dir = os.path.join(output_dir, 'combined_metadata')
    os.makedirs(output_dir, exist_ok=True)

    findings = pd.read_csv(findings_path, compression='gzip')
    findings_name= os.path.basename(findings_path)
    file_name=os.path.splitext(findings_name)[0]
    metadata = pd.read_csv(metadata_csv, compression='gzip')
    split = pd.read_csv(split_csv, compression='gzip')
    meta_findings = metadata.merge(findings, on=["subject_id", "study_id"], how="left")
    extracted_col = split["split"]
    combined_metadata = meta_findings.join(extracted_col)
    #s3 = boto3.resource('s3')
    #my_bucket = s3.Bucket('s3://qtr-ml-projects/edema_quantification/dataset/MIMIC-CXR/files/')
    if dataset_dir:
        img_paths = get_file_list(
            src_dirs=dataset_dir,
            ext_list=['.png', '.jpg', '.jpeg', '.bmp'])
        dicom_id = list()
        for i in img_paths:
            img_name = os.path.basename(i)
            dicom_id.append(os.path.splitext(img_name)[0])

        files = pd.DataFrame(list(zip(dicom_id, img_paths)), columns=['dicom_id', 'file_path'])
        combined_metadata = combined_metadata.merge(files, on=["dicom_id"], how="left")

    #combined_metadata.to_csv(os.path.join(output_dir, f'combined_metadata_{file_name}.csv'), index=1, encoding='utf-8')
    combined_metadata.to_excel(os.path.join(output_dir, f'combined_metadata_{file_name}.xlsx'), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combine_metadata')
    parser.add_argument('--metadata_csv', default='dataset/metadata/mimic-cxr-2.0.0-metadata.csv', type=str)
    parser.add_argument('--findings_path', default='dataset/metadata/mimic-cxr-2.0.0-chexpert.csv', type=str)
    parser.add_argument('--split_csv', default='dataset/metadata/mimic-cxr-2.0.0-split.csv', type=str)
    parser.add_argument('--output_dir', default='dataset/output', type=str)
    parser.add_argument('--dataset_dir', type=str, default=None)
    args = parser.parse_args()
    combine_metadata(
        metadata_csv=args.metadata_csv,
        findings_path=args.findings_path,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
    )
