import caffe
import numpy as np
import cv2
import sys
import time

person_label = 1
max_input_height = 480
max_input_width = 640


def fit_size(image):
    """
    If image's width or height is not multiples of 8,
    add padding to width or height, to make them multiples of 8.
    :param image: input image
    :return: output_image, pad_width, pad_height
    """
    h, w, c = image.shape
    assert c == 3
    pad_h, pad_w = 0, 0
    
    remains = h % 8
    if remains != 0:
        pad_h = 8 - remains

    remains = w % 8
    if remains != 0:
        pad_w = 8 - remains
            
    if pad_h == 0 and pad_w == 0:
        return image, 0, 0
            
    new_h = h + pad_h
    new_w = w + pad_w
    padded_image = np.zeros((new_h, new_w, c), image.dtype)
    padded_image[:h, :w, :] = image
    return padded_image, pad_w, pad_h


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


def predict_5(seg_net, image, thresh=0.5, display=True):
    original_height, original_width, _ = image.shape
    if original_height > max_input_height or original_width > max_input_width:
        h_ratio = max_input_height * 1.0 / original_height
        w_ratio = max_input_width * 1.0 / original_width
        ratio = min(h_ratio, w_ratio)
        rescaled_image = cv2.resize(image, None, fx=ratio, fy=ratio)
    else:
        rescaled_image = image
        
    fit_size_image, pad_w, pad_h = fit_size(rescaled_image)
    input_h, input_w, _ = fit_size_image.shape
    seg_net.blobs['data'].reshape(1, 3, input_h, input_w)
    seg_net.blobs['edge_feature'].reshape(1, 1, input_h, input_w)
    
    for _ in range(10):
        tic = time.time()
        input_data = convert(fit_size_image)
        edge_feature = get_normed_edge_feature(fit_size_image)
        seg_net.blobs['data'].data[...] = input_data
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

    if pad_h > 0:
        seg = seg[:-pad_h, :]
    if pad_w > 0:
        seg = seg[:, :-pad_w]
    
    mask = cv2.resize(seg, (original_width, original_height))
    mask_color = np.zeros((original_height, original_width, 3), np.uint8)
    mask_color[mask == person_label] = [0, 255, 0]
    
    if np.count_nonzero(mask) > 0:
        image[mask == person_label] = cv2.addWeighted(image[mask == person_label], 0.5, mask_color[mask == person_label], 0.5, 0)
    
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
    predict_5(seg_net, img, thresh)

