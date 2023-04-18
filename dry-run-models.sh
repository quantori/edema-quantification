#!/bin/bash

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py" \
&> dry-run-models-logs/vfnet.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py" \
&> dry-run-models-logs/tood.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py" \
&> dry-run-models-logs/gfl.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/paa/paa_r101_fpn_mstrain_3x_coco.py" \
&> dry-run-models-logs/paa.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco.py" \
&> dry-run-models-logs/guided_anchoring.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco.py" \
&> dry-run-models-logs/sabl.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco.py" \
&> dry-run-models-logs/grid_rcnn.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco.py" \
&> dry-run-models-logs/libra_rcnn.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py" \
&> dry-run-models-logs/fcos.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py" \
&> dry-run-models-logs/faster_rcnn.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/fsaf/fsaf_x101_64x4d_fpn_1x_coco.py" \
&> dry-run-models-logs/fsaf.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py" \
&> dry-run-models-logs/cascade_rpn.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/atss/atss_r101_fpn_1x_coco.py" \
&> dry-run-models-logs/atss.log
