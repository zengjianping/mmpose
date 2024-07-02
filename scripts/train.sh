#!/bin/bash

#https://unix.stackexchange.com/questions/87908/how-do-you-empty-the-buffers-and-cache-on-a-linux-system
# free && sync && echo 3 > /proc/sys/vm/drop_caches && free
# sudo bash -c "echo 3 > /proc/sys/vm/drop_caches"

#CONFIG_FILE="configs/body_2d_keypoint/golfpose/rtmpose-m_golfpose-256x192.py"
CONFIG_FILE="configs/body_2d_keypoint/golfpose/rtmpose-m_halpe28-256x192.py"
#CONFIG_FILE="configs/body_2d_keypoint/golfpose/rtmo-m_golfpose-640x640.py"
#CONFIG_FILE="configs/body_2d_keypoint/golfpose/rtmo-m_golfclub-640x640.py"
#CONFIG_FILE="configs/body_2d_keypoint/golfpose/rtmo-t_golfclub-416x416.py"

#export CUDA_LAUNCH_BLOCKING=1
python tools/train.py $CONFIG_FILE --amp

