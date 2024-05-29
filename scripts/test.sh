#!/bin/bash

python tools/test.py \
    configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-golfpose-256x192.py \
    work_dirs/rtmpose-m_8xb512-700e_body8-golfpose-256x192/best_AUC_epoch_100.pth \
    --show

