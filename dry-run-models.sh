#!/bin/bash

python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py" \
&> dry-run-models-logs/vfnet.log


python src/models/mmdetection/tools/train.py \
--config "src/models/mmdetection/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py" \
&> dry-run-models-logs/gfl.log