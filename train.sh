#!/bin/bash

caffe_root=/home/jiajie/openpose/3rdparty/caffe

cd models
${caffe_root}/build/tools/caffe train -solver person_seg_net_2.0_solver.prototxt  -weights person_seg_net_2.0_init_weights.caffemodel  -gpu 1
cd -

