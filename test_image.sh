#!/bin/bash

model=/home/jiajie/video_segment/models/person_seg_net_deploy.prototxt
weights=/home/jiajie/video_segment/models/backup/person_seg_net_iter_2000.caffemodel
image_path=$1

python scripts/predict.py ${model}  ${weights}  ${image_path}
