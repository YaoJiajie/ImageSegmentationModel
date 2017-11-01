import caffe
import numpy as np
from data import fit_size
from data import input_height
from data import input_width
from data import person_label
from data import get_normed_edge_feature
import cv2
import sys


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


def predict(net, image, thresh=0.5):
    original_height, original_width, _ = image.shape
    input_data = convert(image)
    net.blobs['data'].data[...] = input_data
    output = net.forward()
    seg = output['seg_out'][0]

    # # display the pose heat map
    # pose_output = net.blobs['pose_output_8x'].data[0]
    # channels, _, _ = pose_output.shape
    # for ch in range(channels):
    #     heat_map = pose_output[ch]
    #     heat_map_normed = cv2.normalize(heat_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #     heat_map_vis = heat_map_normed.astype(np.uint8)
    #     cv2.imshow('heatmap', heat_map_vis)
    #     cv2.waitKey()

    seg = np.squeeze(seg)
    seg[seg > thresh] = person_label
    seg[seg != person_label] = 0
    seg = seg.astype(np.uint8)

    mask = to_original_scale(seg, (original_height, original_width))
    mask_color = np.zeros((original_height, original_width, 3), np.uint8)
    mask_color[mask == person_label] = [0, 255, 0]

    if np.count_nonzero(mask) > 0:
        image[mask == person_label] = cv2.addWeighted(image[mask == person_label], 0.5, mask_color[mask == person_label], 0.5, 0)

    cv2.imshow('segmentation', image)
    cv2.waitKey()


def predict_2(pose_net, seg_net, image, thresh=0.5):
    original_height, original_width, _ = image.shape
    input_data = convert(image)

    pose_net.blobs['image'].data[...] = input_data
    pose_output = pose_net.forward()
    pose_output = pose_output['net_output']

    # sum all the pose channels and visualize
    pose_output_sum = np.sum(pose_output[0], 0)
    pose_output_sum = cv2.normalize(pose_output_sum, None, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
    pose_output_sum = pose_output_sum.astype(np.uint8)
    pose_output_sum = cv2.resize(pose_output_sum, None, None, fx=8, fy=8)
    cv2.imshow('pose_output_sum', pose_output_sum)

    seg_net.blobs['data'].data[...] = input_data
    seg_net.blobs['pose_output'].data[...] = pose_output
    output = seg_net.forward()
    seg = output['seg_out'][0]

    seg = np.squeeze(seg)
    seg[seg > thresh] = person_label
    seg[seg != person_label] = 0
    seg = seg.astype(np.uint8)

    mask = to_original_scale(seg, (original_height, original_width))
    mask_color = np.zeros((original_height, original_width, 3), np.uint8)
    mask_color[mask == person_label] = [0, 255, 0]

    if np.count_nonzero(mask) > 0:
        image[mask == person_label] = cv2.addWeighted(image[mask == person_label], 0.5,
                                                      mask_color[mask == person_label], 0.5, 0)

    cv2.imshow('segmentation', image)
    cv2.waitKey()


def predict_3(pose_net, seg_net, image, thresh=0.5):
    original_height, original_width, _ = image.shape
    input_data = convert(image)

    fitsize_img = fit_size([image, ])[0]
    cv2.imshow('fitsized_img', fitsize_img)
    cv2.waitKey()
    edge_feature = get_normed_edge_feature(fitsize_img)

    edge_vis = (edge_feature + 0.5) * 255.0
    edge_vis = edge_vis.astype(np.uint8)
    cv2.imshow('edge_input', edge_vis)
    cv2.waitKey()

    pose_net.blobs['image'].data[...] = input_data
    pose_output = pose_net.forward()
    pose_feature_vis = pose_net.blobs['concat_stage2'].data
    pose_output = pose_output['net_output']

    # sum all the pose channels and visualize
    pose_output_sum = np.sum(pose_feature_vis[0], 0)
    pose_output_sum = cv2.normalize(pose_output_sum, None, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
    pose_output_sum = pose_output_sum.astype(np.uint8)
    pose_output_sum = cv2.resize(pose_output_sum, None, None, fx=8, fy=8)
    cv2.imshow('pose_output_sum', pose_output_sum)

    seg_net.blobs['data'].data[...] = input_data
    seg_net.blobs['pose_output'].data[...] = pose_output
    seg_net.blobs['edge_feature'].data[...] = edge_feature[np.newaxis, np.newaxis, :, :]
    output = seg_net.forward()
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
    ver = sys.argv[1]

    if ver == '1':
        net_prototxt = sys.argv[2]
        weights = sys.argv[3]
        image_path = sys.argv[4]
        gpu_id = int(sys.argv[5])
        thresh = float(sys.argv[6])

        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        caffe_net = caffe.Net(net_prototxt, weights, caffe.TEST)
        img = cv2.imread(image_path)
        predict(caffe_net, img, thresh)

    elif ver == '2' or ver == '3':
        pose_prototxt = sys.argv[2]
        pose_weights = sys.argv[3]
        seg_prototxt = sys.argv[4]
        seg_weights = sys.argv[5]
        image_path = sys.argv[6]
        gpu_id = int(sys.argv[7])
        thresh = float(sys.argv[8])

        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        pose_net = caffe.Net(pose_prototxt, pose_weights, caffe.TEST)
        seg_net = caffe.Net(seg_prototxt, seg_weights, caffe.TEST)
        img = cv2.imread(image_path)

        if ver == '2':
            predict_2(pose_net, seg_net, img, thresh)
        else:
            predict_3(pose_net, seg_net, img, thresh)

