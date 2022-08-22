import pandas as pd
import os
import argparse


def combine_metadata(
        metadata_csv:str,
        chexpert_csv:str,
        negbio_csv:str,
        split_csv:str,
        output_dir:str,
) -> None:
    output_dir = os.path.join(output_dir, 'combined_metadata')
    os.makedirs(output_dir, exist_ok=True)

    chexpert=pd.read_csv(chexpert_csv, compression='gzip')
    metadata= pd.read_csv(metadata_csv, compression='gzip')
    negbio=pd.read_csv(negbio_csv, compression='gzip')
    split=pd.read_csv(split_csv, compression='gzip')
    meta_chex = metadata.merge(chexpert, on=["subject_id", "study_id"], how="left")
    extracted_col = split["split"]
    combine_meta_chex=meta_chex.join(extracted_col)
    combine_meta_chex.to_csv(os.path.join(output_dir, 'combine_meta_chex.csv'), encoding='utf-8')
    meta_neg = metadata.merge(negbio, on=["subject_id", "study_id"], how="left")
    combine_meta_neg=meta_neg.join(extracted_col)
    combine_meta_neg.to_csv(os.path.join(output_dir, 'combine_meta_neg.csv'), encoding='utf-8')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combine_metadata')
    parser.add_argument('--metadata_csv', default='dataset/metadata/mimic-cxr-2.0.0-metadata.csv', type=str)
    parser.add_argument('--chexpert_csv', default='dataset/metadata/mimic-cxr-2.0.0-chexpert.csv', type=str)
    parser.add_argument('--negbio_csv', default='dataset/metadata/mimic-cxr-2.0.0-negbio.csv', type=str)
    parser.add_argument('--split_csv', default='dataset/metadata/mimic-cxr-2.0.0-split.csv', type=str)
    parser.add_argument('--output_dir', default='dataset/output', type=str)
    args = parser.parse_args()
    combine_metadata(
        metadata_csv=args.metadata_csv,
        chexpert_csv=args.chexpert_csv,
        negbio_csv=args.negbio_csv,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
    )
