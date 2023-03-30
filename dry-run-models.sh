#!/bin/bash

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/fast_rcnn/fast_rcnn_r101_fpn_2x_coco.py" \
&> dry-run-models-logs/fast_rcnn.log


python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/rpn/rpn_x101_64x4d_fpn_2x_coco.py" \
&> dry-run-models-logs/rpn.log



python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/ssd/ssd512_coco.py" \
&> dry-run-models-logs/ssd.log



python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/retinanet/retinanet_x101_64x4d_fpn_2x_coco.py" \
&> dry-run-models-logs/retinanet.log



python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py" \
&> dry-run-models-logs/yolo.log



python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py" \
&> dry-run-models-logs/cornernet.log



python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco.py" \
&> dry-run-models-logs/grid_rcnn.log



python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco.py" \
&> dry-run-models-logs/guided_anchoring.log



python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco.py" \
&> dry-run-models-logs/free_anchor.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck+head_2x_coco.py" \
&> dry-run-models-logs/reppoints.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py" \
&> dry-run-models-logs/fcos.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco.py" \
&> dry-run-models-logs/libra_rcnn.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/fsaf/fsaf_x101_64x4d_fpn_1x_coco.py" \
&> dry-run-models-logs/fsaf.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py" \
&> dry-run-models-logs/cascade_rpn.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/foveabox/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco.py" \
&> dry-run-models-logs/foveabox.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/atss/atss_r101_fpn_1x_coco.py" \
&> dry-run-models-logs/atss.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco.py" \
&> dry-run-models-logs/sabl.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/paa/paa_r101_fpn_mstrain_3x_coco.py" \
&> dry-run-models-logs/paa.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py" \
&> dry-run-models-logs/deformable_detr.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py" \
&> dry-run-models-logs/yolox.log

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py" \
&> dry-run-models-logs/tood.log