#!/bin/bash

caffe_root=/home/jiajie/openpose/3rdparty/caffe

version=$1

cd models
${caffe_root}/build/tools/caffe train -solver person_seg_net_${version}_solver.prototxt  -weights person_seg_net_${version}_init_weights.caffemodel  -gpu 1
#${caffe_root}/build/tools/caffe train -solver person_seg_net_3.1_solver.prototxt  -snapshot backup/person_seg_net_3.1_iter_180765.solverstate  -gpu 0
cd -


