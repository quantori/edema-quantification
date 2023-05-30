import logging
import os

import fiftyone as fo
import hydra
from fiftyone import ViewField as F
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='visualize_evaluation',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    """Visualize a COCO dataset to verify its correctness.

    Args:
        eval_dataset_path: a json file with prediction and ground truth data
        n_most_common_classes: a number of the most common classes to analyse
    Returns:
        None
    """
    dataset = fo.Dataset.from_json(path_or_str=cfg.eval_dataset_path)
    results = dataset.evaluate_detections(
        'predictions',
        gt_field='ground_truth',
        eval_key='eval',
    )

    counts = dataset.count_values('ground_truth.detections.label')
    classes = sorted(counts, key=counts.get, reverse=True)[: cfg.n_most_common_classes]
    results.print_report(classes=classes)

    print('TP: %d' % dataset.sum('eval_tp'))
    print('FP: %d' % dataset.sum('eval_fp'))
    print('FN: %d' % dataset.sum('eval_fn'))

    view = dataset.sort_by('eval_fp', reverse=True).filter_labels('predictions', F('eval') == 'fp')

    session = fo.launch_app(view=view)

    session.wait()
    dataset.delete()


if __name__ == '__main__':
    main()
