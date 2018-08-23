#!/bin/bash

caffe_root=/home/jiajie/caffe

weights=$1
gpu=$2

cd models
${caffe_root}/build/tools/caffe train -solver person_seg_net_solver.prototxt  -snapshot ${weights}  -gpu ${gpu}





