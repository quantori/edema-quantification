import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import get_model_complexity_info
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (
    collect_env,
    get_device,
    get_root_logger,
    replace_cfg_vals,
    rfnext_init_model,
    setup_multi_processes,
    update_data_root,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--config',
        type=str,
        default='src/models/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        help='path to a train config file',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/coco',
        help='directory to the COCO dataset',
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        default='CocoDataset',
        help='type of the dataset',
    )
    parser.add_argument(
        '--filter-empty-gt',
        action='store_true',
        help='whether to exclude the empty GT images',
    )
    parser.add_argument('--batch-size', type=int, default=None, help='batch size')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='workers to pre-fetch data for each single GPU',
    )
    parser.add_argument('--epochs', default=2, type=int, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=11, help='seed value for reproducible results')
    parser.add_argument(
        '--work-dir',
        default='models/sign_detection',
        help='the dir to save logs and models',
    )
    parser.add_argument(
        '--resume-from',
        help='the checkpoint file to resume from',
    )
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically',
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training',
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)',
    )
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)',
    )
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use ' '(only applicable to non-distributed training)',
    )
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks',
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.',
    )
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.',
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options',
        )
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # ------------------------------------------------- CUSTOM CONFIG --------------------------------------------------
    CLASSES = (
        'Cephalization',
        'Heart',
        'Artery',
        'Bronchus',
        'Kerley',
        'Cuffing',
        'Effusion',
        'Bat',
        'Infiltrate',
    )
    cfg.classes = CLASSES
    cfg.dataset_type = args.dataset_type
    cfg.data_root = args.data_dir

    # Modify num classes of the model in box head
    try:
        cfg.model.bbox_head.num_classes = len(CLASSES)
    except Exception:
        cfg.model.roi_head.bbox_head.num_classes = len(CLASSES)

    cfg.data.train.type = args.dataset_type
    cfg.data.train.classes = cfg.classes
    cfg.data.train.filter_empty_gt = args.filter_empty_gt
    cfg.data.train.ann_file = os.path.join(args.data_dir, 'train', 'labels.json')
    cfg.data.train.img_prefix = os.path.join(args.data_dir, 'train', 'data')

    cfg.data.val.type = args.dataset_type
    cfg.data.val.classes = cfg.classes
    cfg.data.val.ann_file = os.path.join(args.data_dir, 'test', 'labels.json')
    cfg.data.val.img_prefix = os.path.join(args.data_dir, 'test', 'data')

    cfg.data.test.type = args.dataset_type
    cfg.data.test.classes = cfg.classes
    cfg.data.test.ann_file = os.path.join(args.data_dir, 'test', 'labels.json')
    cfg.data.test.img_prefix = os.path.join(args.data_dir, 'test', 'data')

    if args.batch_size is not None:
        cfg.data.samples_per_gpu = args.batch_size

    if args.num_workers is not None:
        cfg.data.workers_per_gpu = args.num_workers

    cfg.evaluation.metric = 'bbox'
    cfg.optimizer.lr = 0.02 / 8  # The original learning rate is set for 8-GPU training.
    cfg.lr_config.warmup = None

    cfg.log_config.interval = 1  # Equal to batch_size

    cfg.evaluation.interval = 1  # Set the evaluation interval
    cfg.checkpoint_config.interval = 1  # Set the checkpoint saving interval

    # Set seed thus the results are more reproducible
    if args.seed is not None:
        cfg.seed = args.seed
        set_random_seed(args.seed, deterministic=False)

    cfg.runner.max_epochs = args.epochs
    cfg.total_epochs = args.epochs

    # Final config used for training
    print(f'Config:\n{cfg.pretty_text}')
    # ------------------------------------------------------------------------------------------------------------------

    if args.auto_scale_lr:
        if (
            'auto_scale_lr' in cfg
            and 'enable' in cfg.auto_scale_lr
            and 'base_batch_size' in cfg.auto_scale_lr
        ):
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn(
                'Can not find "auto_scale_lr" or '
                '"auto_scale_lr.enable" or '
                '"auto_scale_lr.base_batch_size" in your'
                ' configuration file. Please update all the '
                'configuration files to mmdet >= 2.24.1.',
            )

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = osp.join(args.work_dir, osp.splitext(osp.basename(args.config))[0])
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            './work_dirs',
            osp.splitext(osp.basename(args.config))[0],
        )

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn(
            '`--gpus` is deprecated because we only support '
            'single GPU mode in non-distributed training. '
            'Use `gpus=1` now.',
        )
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn(
            '`--gpu-ids` is deprecated, please use `--gpu-id`. '
            'Because we only support single GPU mode in '
            'non-distributed training. Use the first GPU '
            'in `gpu_ids` now.',
        )
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info(
        'Environment info:\n' + dash_line + env_info + '\n' + dash_line,
    )
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(
        f'Set random seed to {seed}, ' f'deterministic: {args.deterministic}',
    )
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'),
    )
    model.init_weights()

    # init rfnext if 'RFSearchHook' is defined in cfg
    rfnext_init_model(model, cfg=cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        assert 'val' in [mode for (mode, _) in cfg.workflow]
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.get(
            'pipeline',
            cfg.data.train.dataset.get('pipeline'),
        )
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES,
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # MLFlow config
    ml_flow_logger_item = [
        logger for logger in cfg.log_config.hooks if 'MlflowLoggerHook' in logger['type']
    ]
    if ml_flow_logger_item:
        ml_flow_logger = ml_flow_logger_item[0]
        ml_flow_logger['exp_name'] = 'Edema'
        ml_flow_logger['params'] = dict(
            cfg=cfg.filename,
            device=cfg.device,
            seed=cfg.seed,
            epochs=args.epochs,
            model_type=cfg.model.type,
            model_backbone_type=cfg.model.backbone.type,
            data_pipeline_img_input_shape=cfg.data.train.pipeline[2].img_scale,
            data_pipeline_train_img_count=len(datasets[0].data_infos),
            base_batch_size=cfg.data.samples_per_gpu,
        )

        # Compute complexity
        try:
            tmp_model = build_detector(
                cfg.model,
                train_cfg=cfg.get('train_cfg'),
                test_cfg=cfg.get('test_cfg'),
            )
            tmp_model.eval()

            if hasattr(tmp_model, 'forward_dummy'):
                tmp_model.forward = tmp_model.forward_dummy
            else:
                raise NotImplementedError(
                    f'FLOPs counter is not currently supported for {tmp_model.__class__.__name__}',
                )

            input_shape = (3, *cfg.data.train.pipeline[2].img_scale)
            flops, params = get_model_complexity_info(tmp_model, input_shape)
            ml_flow_logger['params']['flops_count'] = flops
            ml_flow_logger['params']['params_count'] = params
        except Exception as err:
            print(f'Error: {err}')
            ml_flow_logger['params']['flops_count'] = 'NA'
            ml_flow_logger['params']['params_count'] = 'NA'

    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == '__main__':
    main()
