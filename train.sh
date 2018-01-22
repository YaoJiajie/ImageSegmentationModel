#!/bin/bash

caffe_root=/home/jiajie/caffe

version=$1
weights=$2
gpu=$3

cd models
${caffe_root}/build/tools/caffe train -solver person_seg_net_${version}_solver.prototxt  -snapshot ${weights}  -gpu ${gpu}
#${caffe_root}/build/tools/caffe train -solver person_seg_net_${version}_solver.prototxt  -weights ${weights}  -gpu ${gpu}
cd -


