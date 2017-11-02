import caffe
import numpy as np
from data import fit_size
from data import input_height
from data import input_width
from data import person_label
from data import get_normed_edge_feature
import cv2
import sys
import time


def convert(image):
    image = fit_size([image, ])[0]
    image = image / 256.0
    image = image - 0.5
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


def predict_4(seg_net, image, thresh=0.5):
    for _ in range(10):
        tic = time.time()
        original_height, original_width, _ = image.shape
        input_data = convert(image)
        fitsize_img = fit_size([image, ])[0]
        edge_feature = get_normed_edge_feature(fitsize_img)
        seg_net.blobs['image'].data[...] = input_data
        seg_net.blobs['edge_feature'].data[...] = edge_feature[np.newaxis, np.newaxis, :, :]
        output = seg_net.forward()
        toc = time.time()
        print('seg time total = {:f}.'.format(toc - tic))
    
    seg = output['seg_out'][0]
    seg = np.squeeze(seg)
    seg_heatmap = cv2.normalize(seg, None, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
    seg_heatmap = seg_heatmap.astype(np.uint8)
    cv2.imshow('seg_heatmap', seg_heatmap)
    cv2.waitKey()
    
    seg[seg > thresh] = person_label
    seg[seg != person_label] = 0
    seg = seg.astype(np.uint8)
    
    mask = to_original_scale(seg, (original_height, original_width))
    mask_color = np.zeros((original_height, original_width, 3), np.uint8)
    mask_color[mask == person_label] = [0, 255, 0]

    if np.count_nonzero(mask) > 0:
        image[mask == person_label] = cv2.addWeighted(image[mask == person_label], 0.5,
                                                      mask_color[mask == person_label], 0.5, 0)
    
    # cv2.imshow('segmentation', image)
    # cv2.waitKey()
    cv2.imwrite('seg.png', image)


if __name__ == '__main__':
    device = sys.argv[1]
    seg_prototxt = sys.argv[2]
    seg_weights = sys.argv[3]
    image_path = sys.argv[4]
    
    if device == 'gpu':
        gpu_id = int(sys.argv[5])
        thresh = float(sys.argv[6])
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    else:
        thresh = float(sys.argv[5])
        
    seg_net = caffe.Net(seg_prototxt, seg_weights, caffe.TEST)
    img = cv2.imread(image_path)
    predict_4(seg_net, img, thresh)


    
        
