#!/bin/bash

WORK_MODE=0
[[ $# -eq 1 ]] && WORK_MODE=$1


DATA_DIR="/data/ModelTrainData/PoseData"
POSE_CONFIG="configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"
POSE_WEIGHT="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"

if [ $WORK_MODE == 0 ]; then
python demo/process_dataset.py --work-mode 0 \
    --inputs tests/data/crowdpose \
    --pose2d $POSE_CONFIG --pose2d-weights $POSE_WEIGHT \
    --vis-out-dir vis_results/crowdpose
fi

if [ $WORK_MODE == 1 ]; then
python demo/process_dataset.py --work-mode 1 \
    --inputs /data/ModelTrainData \
    --input-path tests/data/golfpose/person_keypoints_default.json \
    --output-path tests/data/golfpose/person_keypoints_result.json \
    --pose2d $POSE_CONFIG --pose2d-weights $POSE_WEIGHT 
fi


