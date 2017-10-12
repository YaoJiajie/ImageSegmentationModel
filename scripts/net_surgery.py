import caffe
import sys


def surgery(vgg16_prototxt, vgg_weights, open_pose_weights, person_seg_net_prototxt):
    caffe.set_mode_cpu()
    # vgg16_net = caffe.Net(vgg16_prototxt, vgg_weights, caffe.TEST)
    person_seg_net = caffe.Net(person_seg_net_prototxt, open_pose_weights, caffe.TEST)

    # copy weights
    # layer_names = ['conv1_1', 'conv1_2']
    # layer_names += ['conv2_1', 'conv2_2']
    # layer_names += ['conv3_1', 'conv3_2', 'conv3_3']
    # layer_names += ['conv4_1', 'conv4_2', 'conv4_3']
    # layer_names += ['conv5_1', 'conv5_2', 'conv5_3']

    # for layer_name in layer_names:
    #     person_seg_net.params['vgg_' + layer_name] = vgg16_net.params[layer_name]

    person_seg_net.save('person_seg_net_init_weights.caffemodel')


if __name__ == '__main__':
    vgg16_prototxt_path = sys.argv[1]
    vgg16_weights_path = sys.argv[2]
    open_pose_weights_path = sys.argv[3]
    seg_net_prototxt_path = sys.argv[4]
    surgery(vgg16_prototxt_path, vgg16_weights_path, open_pose_weights_path, seg_net_prototxt_path)

