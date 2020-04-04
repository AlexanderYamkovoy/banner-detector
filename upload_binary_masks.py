import cv2 as cv
import numpy as np
import os
import json
import zlib
import base64


def binary_masks(folder_path, binary_path, show_test=False):
    frames_list = next(os.walk(folder_path))[2]
    for frame in frames_list:
        if frame == '.DS_Store':
            continue
        frame_name = frame.split('.')[0]
        img = cv.imread(folder_path + '/' + frame)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        np.save(binary_path + frame_name, mask)

    if show_test:
        test = np.load(binary_path + 'empty0_1.npy')
        cv.imshow('test', test)
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()


def base64_2_mask(folder_path, y_train_path):
    json_list = next(os.walk(folder_path))[2]
    for json_ in json_list:

        if json_ == '.DS_Store':
            continue

        filename = json_.split('.')[0]
        with open(folder_path + '/' + json_) as f:
            file = json.load(f)
            data = file['objects'][0]['bitmap']['data']
            origin = file['objects'][0]['bitmap']['origin']
            binary_mask = np.zeros((256, 256), dtype=np.uint8)

            z = zlib.decompress(base64.b64decode(data))
            n = np.frombuffer(z, np.uint8)
            mask = cv.imdecode(n, cv.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
            binary_mask[origin[1]:(origin[1] + mask.shape[0]), origin[0]:(origin[0] + mask.shape[1])] = mask

            np.save(y_train_path + filename, binary_mask)


f_path = ''
b_path = ''
ann_path = ''

# binary_masks(f_path, b_path)
# base64_2_mask(ann_path, b_path)

