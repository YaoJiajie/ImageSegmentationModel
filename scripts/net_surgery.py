import caffe
import sys


# Used to create init weights for training (copy open-pose's weights to our network)
def surgery(open_pose_weights, person_seg_net_prototxt):
    caffe.set_mode_cpu()
    person_seg_net = caffe.Net(person_seg_net_prototxt, open_pose_weights, caffe.TRAIN)
    person_seg_net.save('person_seg_net_init_weights.caffemodel')


if __name__ == '__main__':
    open_pose_weights_path = sys.argv[1]
    seg_net_prototxt_path = sys.argv[2]
    surgery(open_pose_weights_path, seg_net_prototxt_path)
