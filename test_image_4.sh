#!/bin/bash

model=$1
weights=$2
image_path=$3
gpu=$4
thres=$5

python scripts/predict4.py 'gpu' ${model}  ${weights}  ${image_path} ${gpu} ${thres}


