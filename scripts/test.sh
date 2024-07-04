#!/bin/bash

#CONFIG_FILE="configs/body_2d_keypoint/golfpose/rtmpose-m_golfpose-256x192.py"
#CHECKPOINT_FILE="work_dirs/rtmpose-m_golfpose-256x192/best_checkpoint_v2/best_AUC_epoch_100.pth"

CONFIG_FILE=projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py
#CONFIG_FILE=configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py
CHECKPOINT_FILE=https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth

CONFIG_FILE=demo/mmdetection_cfg/rtmdet_nano_320-8xb32_coco-person.py
CHECKPOINT_FILE=https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth

CONFIG_FILE=configs/body_2d_keypoint/rtmo/body7/rtmo-t_8xb32-600e_body7-416x416.py
CHECKPOINT_FILE=https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-t_8xb32-600e_body7-416x416-f48f75cb_20231219.pth

python tools/test.py \
    $CONFIG_FILE $CHECKPOINT_FILE
