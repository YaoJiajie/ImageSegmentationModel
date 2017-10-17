#!/bin/bash

model=$1
weights=$2
image_path=$3
gpu=1

python scripts/predict.py ${model}  ${weights}  ${image_path} ${gpu}

