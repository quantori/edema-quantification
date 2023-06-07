import argparse
import copy
import os
import json
import os.path as osp
import time
import warnings
from pathlib import Path

import mlflow
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
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
        choices=[
            'src/models/mmdetection/configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py',
            'src/models/mmdetection/configs/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py',
            'src/models/mmdetection/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py',
            'src/models/mmdetection/configs/paa/paa_r101_fpn_mstrain_3x_coco.py',
            'src/models/mmdetection/configs/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco.py',
            'src/models/mmdetection/configs/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco.py',
            'src/models/mmdetection/configs/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco.py',
            'src/models/mmdetection/configs/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco.py',
            'src/models/mmdetection/configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py',
            'src/models/mmdetection/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py',
            'src/models/mmdetection/configs/fsaf/fsaf_x101_64x4d_fpn_1x_coco.py',
            'src/models/mmdetection/configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py',
            'src/models/mmdetection/configs/atss/atss_r101_fpn_1x_coco.py',
            'src/models/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',    # For use in debug and dev
        ],
        help='path to a train config file',
    )
    parser.add_argument('--data-dir', type=str, default='data/coco', help='directory to the COCO dataset')
    parser.add_argument('--dataset-type', type=str, default='CocoDataset', help='type of the dataset')
    # ----------------------------------------------- CUSTOM ARGUMENTS -------------------------------------------------
    parser.add_argument('--filter-empty-gt', action='store_true', help='whether to exclude the empty GT images')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size')
    parser.add_argument('--img-size', type=int, nargs='+', default=[1536, 1536], help='input image size')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'RMSprop', 'Adam', 'RAdam'], help='optimizer')
    parser.add_argument('--lr', type=float, default=0.1, help='optimizer learning rate')
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0], help='anchor box ratios')
    parser.add_argument('--use-augmentation', action='store_true', help='use augmentation for the train dataset')
    parser.add_argument('--epochs', default=20, type=int, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=11, help='seed value for reproducible results')
    parser.add_argument('--num-workers', type=int, default=None, help='workers to pre-fetch data for each single GPU')
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--work-dir', default='models/feature_detection', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--auto-resume', action='store_true', help='resume from the latest checkpoint automatically')
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int, help='(Deprecated, please use --gpu-id) number of gpus to use')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='(Deprecated, please use --gpu-id) ids of gpus to use')
    group_gpus.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use')
    parser.add_argument('--diff-seed', action='store_true', help='Whether or not set different seeds for different ranks')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend')
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
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--auto-scale-lr', action='store_true', help='enable automatically scaling LR')
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
    # Set the list of classes in the dataset
    ann_file = os.path.join(args.data_dir, 'train', 'labels.json')
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    categories = coco_data['categories']
    class_names_ = [category['name'] for category in categories]
    class_names = tuple(class_names_)
    cfg.classes = class_names

    # Set num classes of the model in box head
    try:
        cfg.model.bbox_head.num_classes = len(class_names)
    except Exception:
        cfg.model.roi_head.bbox_head.num_classes = len(class_names)

    # Set anchor box ratios
    model_family = str(Path(args.config).parent.name)
    if model_family in ['grid_rcnn', 'libra_rcnn', 'faster_rcnn']:
        cfg.model.rpn_head.anchor_generator['ratios'] = args.ratios
    elif model_family in ['guided_anchoring']:
        cfg.model.rpn_head.approx_anchor_generator['ratios'] = args.ratios
    elif model_family in ['cascade_rpn']:
        cfg.model.rpn_head.stages[0].anchor_generator['ratios'] = args.ratios
    elif model_family in ['tood', 'gfl', 'paa', 'fsaf', 'atss']:
        cfg.model.bbox_head.anchor_generator['ratios'] = args.ratios
    elif model_family in ['sabl']:
        cfg.model.bbox_head.approx_anchor_generator['ratios'] = args.ratios
    else:
        print(f'\n{cfg.model.type} will be using Default anchor_generator ratios\n')

    # Set dataset metadata
    cfg.data_root = args.data_dir
    cfg.dataset_type = args.dataset_type

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

    # Set batch size and number of workers
    if args.batch_size is not None:
        cfg.data.samples_per_gpu = args.batch_size

    if args.num_workers is not None:
        cfg.data.workers_per_gpu = args.num_workers

    # Set optimizer and learning rate
    if args.optimizer == 'SGD':
        cfg.optimizer = dict(
            type='SGD',
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.0001,
        )
    elif args.optimizer == 'RMSprop':
        cfg.optimizer = dict(
            type='RMSprop',
            lr=args.lr,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0,
        )
    elif args.optimizer == 'Adam':
        cfg.optimizer = dict(
            type='Adam',
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0001,
        )
    elif args.optimizer == 'RAdam':
        cfg.optimizer = dict(
            type='RAdam',
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0001,
        )
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    # Set learning rate scheme
    # https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py
    cfg.lr_config = dict(
        policy='CosineAnnealing',
        warmup='linear',
        warmup_iters=int(0.25 * args.epochs),
        warmup_ratio=0.1,
        min_lr=args.lr / 100,
        warmup_by_epoch=True,
        by_epoch=True,
    )

    # Set the evaluation metric
    cfg.evaluation.metric = 'bbox'

    # Modify log interval (equal to batch_size)
    cfg.log_config.interval = 1

    # Set the evaluation interval
    cfg.evaluation.interval = 1

    # Set the checkpoint saving interval
    cfg.checkpoint_config.interval = 1

    # Set seed thus the results are more reproducible
    if args.seed is not None:
        cfg.seed = args.seed
        set_random_seed(args.seed, deterministic=False)

    # Set training by epoch
    cfg.total_epochs = args.epochs
    cfg.runner = dict(
        type='EpochBasedRunner',
        max_epochs=args.epochs,
    )

    # Augmentation settings
    # Docs: https://mmdetection.readthedocs.io/en/v2.15.1/api.html
    if args.use_augmentation:
        cfg.train_pipeline = [
            dict(
                type='LoadImageFromFile',
                to_float32=True,
            ),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
            ),
            dict(
                type='Resize',
                img_scale=cfg.data.train.pipeline[2].img_scale,
                multiscale_mode='range',
                ratio_range=[0.75, 1],
                keep_ratio=True,
                bbox_clip_border=True,
            ),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3,
                bbox_clip_border=True,
            ),
            dict(
                type='RandomFlip',
                direction='horizontal',
                flip_ratio=0.5,
            ),
            dict(
                type='Rotate',
                level=1,
                max_rotate_angle=20,
                prob=0.2,
            ),
            dict(
                type='Translate',
                level=1,
                max_translate_offset=int(0.1 * min(cfg.data.train.pipeline[2].img_scale)),
                prob=0.2,
            ),
            dict(
                type='EqualizeTransform',
                prob=0.2,
            ),
            dict(
                type='BrightnessTransform',
                level=1,
                prob=0.2,
            ),
            dict(
                type='ContrastTransform',
                level=1,
                prob=0.2,
            ),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            ),
            dict(
                type='Pad',
                size_divisor=32,
            ),
            dict(
                type='DefaultFormatBundle',
            ),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'],
            ),
        ]

    for pipeline in [
        cfg.data.train.pipeline,
        cfg.data.val.pipeline,
        cfg.data.test.pipeline,
        cfg.train_pipeline,
        cfg.test_pipeline,
    ]:
        for step in pipeline:
            if 'img_scale' in step:
                step['img_scale'] = tuple(args.img_size)

    # Final config used for training and testing
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
    timestamp = time.strftime('%d%m_%H%M%S', time.localtime())
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = osp.join(args.work_dir, f'{cfg.model.type}_{timestamp}')
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
        mlflow.set_experiment('Edema')
        run_name = f'{cfg.model.type}_{timestamp}'
        mlflow.set_tag('mlflow.runName', run_name)
        ml_flow_logger['params'] = dict(
            train_images=len(os.listdir(cfg.data.train.img_prefix)),
            test_images=len(os.listdir(cfg.data.test.img_prefix)),
            classes=model.CLASSES,
            num_classes=len(model.CLASSES),
            model=cfg.model.type,
            backbone=cfg.model.backbone.type,
            img_size=cfg.data.train.pipeline[2].img_scale,
            batch_size=cfg.data.samples_per_gpu,
            optimizer=cfg.optimizer.type,
            lr=cfg.optimizer.lr,
            epochs=args.epochs,
            seed=cfg.seed,
            use_augmentation=args.use_augmentation,
            device=cfg.device,
            cfg=cfg.filename,
        )

        # Compute complexity
        # try:
        #     tmp_model = build_detector(
        #         cfg.model,
        #         train_cfg=cfg.get('train_cfg'),
        #         test_cfg=cfg.get('test_cfg'),
        #     )
        #     tmp_model.eval()
        #
        #     if hasattr(tmp_model, 'forward_dummy'):
        #         tmp_model.forward = tmp_model.forward_dummy
        #     else:
        #         raise NotImplementedError(
        #             f'FLOPs counter is not currently supported for {tmp_model.__class__.__name__}',
        #         )
        #
        #     input_shape = (3, *cfg.data.train.pipeline[2].img_scale)
        #     flops, params = get_model_complexity_info(tmp_model, input_shape)
        #     ml_flow_logger['params']['flops'] = flops
        #     ml_flow_logger['params']['params'] = params
        # except Exception as err:
        #     print(f'Error: {err}')
        #     ml_flow_logger['params']['flops'] = 'NA'
        #     ml_flow_logger['params']['params'] = 'NA'

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
