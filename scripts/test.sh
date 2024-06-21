#!/bin/bash

CONFIG_FILE="configs/body_2d_keypoint/golfpose/rtmpose-m_golfpose-256x192.py"
MODEL_FILE="work_dirs/rtmpose-m_golfpose-256x192/best_checkpoint_v2/best_AUC_epoch_100.pth"
SHOW_DIR="evaluations/"

python tools/test.py \
    $CONFIG_FILE $MODEL_FILE \
    --show --show-dir $SHOW_DIR

