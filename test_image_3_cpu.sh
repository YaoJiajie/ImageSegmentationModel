#!/bin/bash

pose_model=models/openpose_pose.prototxt
pose_weights=models/pose_iter_440000.caffemodel

model=$1
weights=$2
image_path=$3
thres=$4

python scripts/predict_cpu.py ${pose_model}  ${pose_weights}  ${model}  ${weights}  ${image_path} ${thres}




