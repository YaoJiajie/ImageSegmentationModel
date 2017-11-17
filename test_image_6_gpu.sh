#!/bin/bash

model=$1
weights=$2
image_path=$3
thres=$4

python scripts/predict6.py 'gpu' ${model}  ${weights}  ${image_path}  0  ${thres}





