#!/bin/bash

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py" \
&> dry-run-models-logs/sparse_rcnn.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py" \
&> dry-run-models-logs/faster_rcnn.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py" \
&> dry-run-models-logs/hrnet.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py" \
&> dry-run-models-logs/cascade_rcnn.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/resnest/faster_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py" \
&> dry-run-models-logs/resnest_faster_rcnn.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/resnest/cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py" \
&> dry-run-models-logs/resnest_cascade_rcn.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py" \
&> dry-run-models-logs/dcn.log