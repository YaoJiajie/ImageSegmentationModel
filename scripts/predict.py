import caffe
import numpy as np
from data import fit_size
from data import input_height
from data import input_width
from data import person_label
import cv2
import sys


def convert(image):
    image = fit_size([image, ])[0]
    image = np.transpose(image, (2, 0, 1))
    data = image.astype(np.float32)
    data = data[np.newaxis, :, :, :]
    return data


def to_original_scale(seg, shape):
    h, w = shape[0], shape[1]
    if h == input_height and w == input_width:
        return seg
    h_ratio = input_height * 1.0 / h
    w_ratio = input_width * 1.0 / w
    ratio = min(h_ratio, w_ratio)
    dst_h = int(h * ratio)
    dst_w = int(w * ratio)
    h_offset = (input_height - dst_h) / 2
    w_offset = (input_width - dst_w) / 2
    roi = seg[h_offset:h_offset + dst_h, w_offset:w_offset + dst_w]
    return cv2.resize(roi, (w, h), interpolation=cv2.INTER_CUBIC)


def predict(net, image):
    original_height, original_width, _ = image.shape
    input_data = convert(image)
    net.blobs['data'].data[...] = input_data
    output = net.forward()
    seg = output['seg_out'][0]
    seg = np.argmax(seg, 0)
    seg = seg.astype(np.uint8)
    mask = to_original_scale(seg, (original_height, original_width))
    mask_color = np.zeros((original_height, original_width, 3), np.uint8)
    mask_color[mask == person_label] = [0, 255, 0]

    image[mask == person_label] = cv2.addWeighted(image[mask == person_label], 0.5, mask_color[mask == person_label], 0.5, 0)
    cv2.imshow('segmentation', image)
    cv2.waitKey()
    cv2.imwrite('segmentation.png', image)


if __name__ == '__main__':
    net_prototxt = sys.argv[1]
    weights = sys.argv[2]
    image_path = sys.argv[3]
    caffe.set_mode_gpu()
    caffe_net = caffe.Net(net_prototxt, weights, caffe.TEST)
    img = cv2.imread(image_path)
    # cv2.imshow('input', img)
    # cv2.waitKey()
    predict(caffe_net, img)
