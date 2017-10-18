#!/bin/bash

pose_model=models/openpose_pose.prototxt
pose_weights=models/pose_iter_440000.caffemodel

model=$1
weights=$2
image_path=$3
gpu=$4
thres=$5

python scripts/predict.py 2 ${pose_model}  ${pose_weights}  ${model}  ${weights}  ${image_path} ${gpu} ${thres}

