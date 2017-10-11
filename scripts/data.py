import sys
import os
import numpy as np
import cv2
from coco_api.pycocotools.coco import COCO
from coco_api.pycocotools import mask as mask_utils
import lmdb
import caffe


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


if __name__ == '__main__':
    coc_images_dir = sys.argv[1]
    annot_file_path = sys.argv[2]
    out_dir = sys.argv[3]
    create_lmdb(coc_images_dir, annot_file_path, out_dir)
