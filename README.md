# Person Segment

## Dependencies

    1. Cuda9.0+
    2. pycoco-api

        $ git clone https://github.com/YaoJiajie/cocoapi.git
        $ cd cocoapi/PythonAPI
        $ python setup install --prefix ~/.local
       
    3. caffe
    
        $ git clone https://github.com/BVLC/caffe.git
        $ cd caffe
        $ mkdir build && cd build
        $ cmake ..
        $ make -j
        $ make pycaffe #  make sure pycaffe is made successful
        $ export PYTHONPATH=<CAFFE_SRC_PATH>/python:$PYTHONPATH

## Prepare Data

    1. Download CoCo2017 data.
    2. Create BGR image lmdb & label lmdb

        $ python data.py data <coco_path>/train2017  <coco_path>/annotations/instances_train2017.json  <dst_path>  # make sure dst_path exists
        $ python data.py data <coco_path>/val2017  <coco_path>/annotations/instances_val2017.json  <dst_path>  # make sure dst_path exists

    3. Create Edge feature lmdb

        $ python data.py edge  <path_to_train_bgr_lmdb>  <path_to_output_train_edge_lmdb>
        $ python data.py edge  <path_to_val_bgr_lmdb>   <path_to_output_val_edge_lmdb>

## Training

    1. Download OpenPose weights.

        $ git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
        $ cd openpose/models
        $ ./getModels.sh
	
    2. Create init weights from OpenPose

        $ python net_surgery.py <path_to_openpose.caffemodel>  <path_to_person_segment_net.prototxt>

    3. Modify the paths in network prototxt (lmdb paths), paths in solver prototxt, paths in train.sh

        $  ./train.sh

    4.  Scripts and prototxt for evaluating mIOU are also provided.
