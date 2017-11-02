#!/bin/bash

model=$1
weights=$2
image_path=$3
gpu=$4
thres=$5

python scripts/predict4.py ${pose_model}  ${pose_weights}  ${model}  ${weights}  ${image_path} ${gpu} ${thres}


