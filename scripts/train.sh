#!/bin/bash

#https://unix.stackexchange.com/questions/87908/how-do-you-empty-the-buffers-and-cache-on-a-linux-system
# free && sync && echo 3 > /proc/sys/vm/drop_caches && free

python tools/train.py \
    configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-golfpose-256x192.py \
    --amp

