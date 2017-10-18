import sys
import os
import numpy as np
import cv2
from coco_api.pycocotools.coco import COCO
from coco_api.pycocotools import mask as mask_utils
import lmdb
import caffe
from caffe.proto import caffe_pb2

coco_format = '{:012d}.jpg'
person_label = 1
input_height = 480
input_width = 640


def parse_annot(coco, image_path, annots):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    polygens = []
    masks = []

    for annot in annots:
        if 'segmentation' in annot:
            if type(annot['segmentation']) is list:
                # polygen
                for seg in annot['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    poly = poly.astype(np.int32)
                    polygens.append(poly)
            else:
                # mask
                t = coco.loadImgs([annot['image_id'], ])[0]
                if type(annot['segmentation']['counts']) == list:
                    rle = mask_utils.frPyObjects([annot['segmentation']], t['height'], t['width'])
                else:
                    rle = [annot['segmentation']]
                m = mask_utils.decode(rle)
                m = np.squeeze(m)
                masks.append(m)

    h, w, _ = image.shape
    label_image = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(label_image, polygens, person_label, cv2.LINE_AA)

    for m in masks:
        label_image[m != 0] = person_label

    image, label_image = fit_size([image, label_image])

    # cv2.imshow('image', image)
    # cv2.imshow('label', label_image)
    # cv2.waitKey()

    return image, label_image


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


def create_lmdb(coco_images_path, annot_file, lmdb_dir):
    coco = COCO(annot_file)
    person_cat_id = coco.getCatIds(catNms=['person', ])[0]
    img_ids = coco.getImgIds(catIds=[person_cat_id, ])
    images = coco.loadImgs(img_ids)
    annot_type, _ = os.path.splitext(os.path.basename(annot_file))

    data_lmdb_path = os.path.join(lmdb_dir, 'person_' + annot_type + '_data')
    label_lmdb_path = os.path.join(lmdb_dir, 'person_' + annot_type + '_label')
    data_lmdb = lmdb.open(data_lmdb_path, map_size=int(1e12))
    data_txn = data_lmdb.begin(write=True)
    label_lmdb = lmdb.open(label_lmdb_path, map_size=int(1e12))
    label_txn = label_lmdb.begin(write=True)
    sample_count = 0
    total_count = len(images)

    for image in images:
        img_id = image['id']
        image_filename = coco_format.format(img_id)
        db_key, _ = os.path.splitext(image_filename)

        image_path = os.path.join(coco_images_path, image_filename)
        annot_ids = coco.getAnnIds(imgIds=[img_id, ], catIds=[person_cat_id, ], iscrowd=None)
        anns = coco.loadAnns(annot_ids)
        image_data, label_data = parse_annot(coco, image_path, anns)
        image_data = np.transpose(image_data, (2, 0, 1))
        label_data = label_data[np.newaxis, :, :]

        # print(image_data.shape)
        # print(label_data.shape)

        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        data_datum = caffe.io.array_to_datum(image_data)
        data_txn.put(db_key, data_datum.SerializeToString())

        label_datum = caffe.io.array_to_datum(label_data)
        label_txn.put(db_key, label_datum.SerializeToString())

        sample_count += 1

        if sample_count % 1000 == 0:
            data_txn.commit()
            label_txn.commit()
            data_txn = data_lmdb.begin(write=True)
            label_txn = label_lmdb.begin(write=True)
            print('{:d}/{:d} samples being processed.'.format(sample_count, total_count))

    data_txn.commit()
    label_txn.commit()
    data_lmdb.close()
    label_lmdb.close()
    print('Total {:d}/{:d} samples beging processed.'.format(sample_count, total_count))


def create_pose_lmdb(openpose_prototxt, openpose_weights, output_lmdb_path, sample_num):
    caffe.set_mode_gpu()
    caffe.set_device(1)
    pose_net = caffe.Net(openpose_prototxt, openpose_weights, caffe.TEST)
    count = 0
    pose_lmdb = lmdb.open(output_lmdb_path, map_size=int(1e12))
    pose_txn = pose_lmdb.begin(write=True)

    while count < sample_num:
        output = pose_net.forward()
        pose_features = output['net_output']

        for pose_feature in pose_features:
            db_key = '{:012d}'.format(count)
            datum = caffe.io.array_to_datum(pose_feature)
            pose_txn.put(db_key, datum.SerializeToString())
            count += 1

            if count % 1000 == 0:
                pose_txn.commit()
                pose_txn = pose_lmdb.begin(write=True)
                print('{:d}/{:d} samples being processed.'.format(count, sample_num))
                sys.stdout.flush()

            if count >= sample_num:
                break

        # print(pose_features.shape)
        # # visualize the first image's first channel:
        # pose_feature_vis = pose_features[0][0]
        # pose_feature_vis = cv2.normalize(pose_feature_vis, None, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
        # pose_feature_vis = pose_feature_vis.astype(np.uint8)
        # cv2.imshow('pose_feature', pose_feature_vis)
        # cv2.waitKey()

    pose_txn.commit()
    pose_lmdb.close()
    print('Total {:d}/{:d} samples beging processed.\n'.format(count, sample_num))


def create_edge_lmdb(bgr_lmdb_path, output_lmdb_path):
    bgr_lmdb = lmdb.open(bgr_lmdb_path)
    bgr_txn = bgr_lmdb.begin()
    bgr_cursor = bgr_txn.cursor()
    datum = caffe_pb2.Datum()
    count = 0

    feature_lmdb = lmdb.open(output_lmdb_path, map_size=int(1e12))
    feature_txn = feature_lmdb.begin(write=True)

    for db_key, value in bgr_cursor:
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)

        im = data.astype(np.uint8)  # c * h * w
        im = np.transpose(im, (1, 2, 0))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, None, 3)
        grad_x = np.abs(grad_x)
        grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, None, 3)
        grad_y = np.abs(grad_y)
        total_grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        total_grad = cv2.normalize(total_grad, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        total_grad -= 0.5

        # grad_vis = cv2.normalize(total_grad, None, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
        # grad_vis = grad_vis.astype(np.uint8)
        # cv2.imshow('image', im)
        # cv2.imshow('grad', grad_vis)
        # cv2.waitKey()

        total_grad = total_grad[np.newaxis, :, :]
        feature_datum = caffe.io.array_to_datum(total_grad)
        feature_txn.put(db_key, feature_datum.SerializeToString())
        count += 1

        if count % 1000 == 0:
            feature_txn.commit()
            feature_txn = feature_lmdb.begin(write=True)
            print('{:d} samples being processed.'.format(count))
            sys.stdout.flush()

    feature_txn.commit()
    feature_lmdb.close()
    bgr_lmdb.close()
    print('Total {:d} samples being processed.'.format(count))


if __name__ == '__main__':
    # create_lmdb(sys.argv[1], sys.argv[2], sys.argv[3])
    # create_pose_lmdb(sys.argv[1], sys.argv[2], sys.argv[3], 64115)
    create_edge_lmdb(sys.argv[1], sys.argv[2])
