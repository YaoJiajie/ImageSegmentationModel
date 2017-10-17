#!/bin/bash

model=$1
weights=$2
image_path=$3
gpu=$4
thres=$5

python scripts/predict.py  1  ${model}  ${weights}  ${image_path} ${gpu} ${thres}
