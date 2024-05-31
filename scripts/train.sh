#!/bin/bash

#https://unix.stackexchange.com/questions/87908/how-do-you-empty-the-buffers-and-cache-on-a-linux-system
# free && sync && echo 3 > /proc/sys/vm/drop_caches && free

#CONFIG_FILE="configs/body_2d_keypoint/golfpose/rtmpose-m_golfpose-256x192.py"
CONFIG_FILE="configs/body_2d_keypoint/golfpose/rtmo-m_golfpose-640x640.py"

python tools/train.py $CONFIG_FILE --amp

