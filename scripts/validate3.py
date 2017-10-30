import caffe
import os
import sys


def validate(net, sample_num=2693, batch_size=4):
    count = 0
    mean_loss = 0.0

    while count < sample_num:
        output = net.forward()
        loss = output['loss']
        mean_loss += loss
        count += batch_size
        print(count)
        if count >= sample_num:
            break

    mean_loss /= count
    return mean_loss


def validate_all(seg_net_prototxt, weights_dir, interval=2000, max_iter=500000):
    weights_format = 'person_seg_net_3.1_iter_{:d}.caffemodel'
    iters = []
    losses = []
    output_file = open('validate.txt', 'w')

    for iter_idx in range(0, max_iter, interval):
        weights_path = weights_format.format(iter_idx)
        weights_path = os.path.join(weights_dir, weights_path)

        if not os.path.exists(weights_path):
            continue

        net = caffe.Net(seg_net_prototxt, weights_path, caffe.TEST)
        loss = validate(net)
        iters.append(iter_idx)
        losses.append(loss)
        result_str = 'iter: {:d}, loss: {:f}'.format(iter_idx, loss)
        print(result_str)
        output_file.write(result_str + '\n')


if __name__ == '__main__':
    validate_all(sys.argv[1], sys.argv[2])
