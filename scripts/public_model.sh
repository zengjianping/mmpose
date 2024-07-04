#!/bin/bash

MODEL_DIR="work_dirs/rtmo-t_halpe28-416x416"
INPUT_MODEL="$MODEL_DIR/epoch_60.pth"
OUTPUT_MODEL="$MODEL_DIR/epoch_60_publish.pth"

python tools/misc/publish_model.py $INPUT_MODEL $OUTPUT_MODEL
