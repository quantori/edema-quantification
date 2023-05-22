import logging
import os

import fiftyone as fo
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='visualize_coco_sigh_detections',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    """Visualize a COCO dataset to verify its correctness.

    Args:
        dataset_dir: a directory with COCO data, including training and test subsets
        subset: a name of subset i.e. train or test
        dataset_name: a name for the dataset
        max_samples: a maximum number of samples to import. By default, all samples are imported
        shuffle: whether to randomly shuffle the order in which the samples are imported
        seed: a random seed to use when shuffling
    Returns:
        None
    """
    subset_dir = os.path.join(cfg.dataset_dir, cfg.subset)
    try:
        dataset = fo.Dataset.from_dir(
            dataset_dir=subset_dir,
            dataset_type=fo.types.COCODetectionDataset,
            overwrite=True,
            persistent=False,
            max_samples=cfg.max_samples,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
        )
    except ValueError:
        dataset = fo.load_dataset(cfg.dataset_name)
    session = fo.launch_app(dataset)
    session.wait()
    dataset.delete()


if __name__ == '__main__':
    main()
