#!/bin/bash

caffe_root=/home/jiajie/openpose/3rdparty/caffe

cd models
${caffe_root}/build/tools/caffe train -solver person_seg_net_1.1_solver.prototxt  -weights person_seg_net_1.1_init_weights.caffemodel  -gpu 0
cd -

