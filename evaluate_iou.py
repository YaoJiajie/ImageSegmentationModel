import caffe
import numpy as np
import sys


def evaluate(model_path, weights_path, sample_num, gpu_id=0):
    #  assume the eval-network output a 2-channel map, first channel is prediction, second channel is label.
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(model_path, weights_path, caffe.TEST)
    count = 0
    sum_iou = 0

    while count < sample_num:
        outputs = net.forward()
        outputs = outputs['eval_output']
        for output_pair in outputs:
            predict = output_pair[0]
            label = output_pair[1]
            predict[predict > 0.5] = 255.0
            predict[predict <= 0.5] = 0.0
            predict = predict.astype(np.uint8)
            label[label != 0] = 255.0
            label = label.astype(np.uint8)
            intersection = np.bitwise_and(predict, label)
            union = np.bitwise_or(predict, label)
            iou = np.count_nonzero(intersection) * 1.0 / np.count_nonzero(union)
            sum_iou += iou
            count += 1
            print(count)
        print('current m-iou = {:f}'.format(sum_iou / count))
        
    mean_iou = sum_iou / count
    print('Mean IOU = {:f}'.format(mean_iou))


if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2], int(sys.argv[3]))

