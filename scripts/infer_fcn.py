import numpy as np
import caffe
import sys
import cv2
import time


max_input_height = 480
max_input_width = 640
net_prototxt = '/home/jiajie/FCN_coco_person/person_seg_v1_deploy.prototxt'
net_weights = '/home/jiajie/FCN_coco_person/person_seg_v1_iter_50000.caffemodel'
use_gpu = True


if __name__ == '__main__':
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = cv2.imread(sys.argv[1])
    
    height, width, _ = im.shape
    if height > max_input_height or width > max_input_width:
        h_rescale = max_input_height * 1.0 / height
        w_rescale = max_input_width * 1.0 / width
        rescale = min(h_rescale, w_rescale)
        im = cv2.resize(im, None, fx=rescale, fy=rescale)

    cv2.imshow('input', im)
    cv2.waitKey()
    
    in_ = np.array(im, dtype=np.float32)
    
    in_ -= np.array((94.3108,97.2271,107.083))
    in_ = in_.transpose((2,0,1))
    in_ = in_[np.newaxis, :, :, :]
    in_ /= 255.0
    
    if use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(0)
    
    # load net
    net = caffe.Net(net_prototxt, net_weights, caffe.TEST)
    
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(*in_.shape)
    net.blobs['data'].data[...] = in_
    
    for _ in range(10):
        tic = time.time()
        net_output = net.forward()
        toc = time.time()
        print('forward time: ' + str(toc - tic))
        
    out = net_output['prob'][0]
    out = np.squeeze(out)

    #out = cv2.normalize(out, None, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
    
    out[out > 0.5] = 255.0
    out[out <= 0.5] = 0.0
    out = out.astype(np.uint8)


    height, width = out.shape
    mask = out
    mask_color = np.zeros((height, width, 3), np.uint8)
    mask_color[mask != 0] = [0, 255, 0]
    
    if np.count_nonzero(mask) > 0:
        im[mask != 0] = cv2.addWeighted(im[mask != 0], 0.5, mask_color[mask != 0], 0.5, 0)
    

    cv2.imshow('segmentation', im)
    cv2.waitKey()
    cv2.imwrite('seg_fcn.png', im)


