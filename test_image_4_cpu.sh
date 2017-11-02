#!/bin/bash

model=$1
weights=$2
image_path=$3
thres=$5

python scripts/predict4.py 'cpu'  ${pose_model}  ${pose_weights}  ${model}  ${weights}  ${image_path} ${thres}



