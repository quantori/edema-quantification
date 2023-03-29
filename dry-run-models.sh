#!/bin/bash
python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/vfnet/vfnet_x101_64x4d_fpn_mstrain_2x_coco.py" \
&> dry-run-models-logs/vfnet_x101_64x4d_fpn_mstrain_2x_coco.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/htc/htc_x101_64x4d_fpn_16x1_20e_coco.py" \
&> dry-run-models-logs/htc_x101_64x4d_fpn_16x1_20e_coco.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/detectors/detectors_htc_r50_1x_coco.py" \
&> dry-run-models-logs/detectors_htc_r50_1x_coco.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py" \
&> dry-run-models-logs/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/resnest/cascade_mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco.py " \
&> dry-run-models-logs/cascade_mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/scnet/scnet_x101_64x4d_fpn_20e_coco.py" \
&> dry-run-models-logs/scnet_x101_64x4d_fpn_20e_coco.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/res2net/htc_r2_101_fpn_20e_coco.py" \
&> dry-run-models-logs/htc_r2_101_fpn_20e_coco.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py" \
&> dry-run-models-logs/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py" \
&> dry-run-models-logs/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.log