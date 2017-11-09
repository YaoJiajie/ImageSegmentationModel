import caffe
import numpy as np
# from data import fit_size
# from data import input_height
# from data import input_width
# from data import person_label
# from data import get_normed_edge_feature
import cv2
import sys
import time


person_label = 1
input_height = 480
input_width = 640


def fit_size(images):
    h, w, _ = images[0].shape
    if h == input_height and w == input_width:
        return images

    h_ratio = input_height * 1.0 / h
    w_ratio = input_width * 1.0 / w

    ratio = min(h_ratio, w_ratio)
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    new_imgs = []
    # print('{:d},{:d} --> {:d},{:d}'.format(h, w, new_h, new_w))

    for img in images:
        if len(img.shape) == 2:
            new_img = np.zeros((input_height, input_width), np.uint8)
        else:
            new_img = np.zeros((input_height, input_width, 3), np.uint8)
        new_img[:] = 0
        h_offset = (input_height - new_h) / 2
        w_offset = (input_width - new_w) / 2
        new_img[h_offset:h_offset + new_h, w_offset:w_offset + new_w] = cv2.resize(img, (new_w, new_h))
        new_imgs.append(new_img)
    return new_imgs


def get_normed_edge_feature(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, None, 3)
    grad_x = np.abs(grad_x)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, None, 3)
    grad_y = np.abs(grad_y)
    total_grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    total_grad = cv2.normalize(total_grad, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    total_grad -= 0.5
    return total_grad


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


def predict_4(seg_net, image, thresh=0.5, display=False):
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

    if display:
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
        image[mask == person_label] = cv2.addWeighted(image[mask == person_label], 0.5, mask_color[mask == person_label], 0.5, 0)
    
    if display:
        cv2.imshow('segmentation', image)
        cv2.waitKey()
    cv2.imwrite('seg.png', image)


def predict_4_1(seg_net, image, thresh=0.5, display=False):
    for _ in range(10):
        tic = time.time()
        original_height, original_width, _ = image.shape
        input_data = convert(image)
        fitsize_img = fit_size([image, ])[0]
        edge_feature = get_normed_edge_feature(fitsize_img)
        edge_feature = cv2.resize(edge_feature, None, fx=1.0/8, fy=1.0/8)
        seg_net.blobs['image'].data[...] = input_data
        seg_net.blobs['edge_feature'].data[...] = edge_feature[np.newaxis, np.newaxis, :, :]
        output = seg_net.forward()
        toc = time.time()
        print('seg time total = {:f}.'.format(toc - tic))

    seg = output['seg_out'][0]
    seg = np.squeeze(seg)
    seg = cv2.resize(seg, None, fx=8, fy=8)

    if display:
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

    if display:
        cv2.imshow('segmentation', image)
        cv2.waitKey()
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
    # predict_4_1(seg_net, img, thresh)

