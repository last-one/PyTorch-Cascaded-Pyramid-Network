import torch
import torch.utils.data as data
import numpy as np
import shutil
import time
import random
import os
import math
import json
import cv2
import Mytransforms

def read_data_file(file_dir):

    lists = []
    with open(file_dir, 'r') as fp:
        line = fp.readline()
        while line:
            path = line.strip()
            lists.append(path)
            line = fp.readline()

    return lists

def read_json_file(file_dir):
    """
        filename: JSON file

        return: two list: key_points list and centers list
    """
    fp = open(file_dir)
    data = json.load(fp)
    kpts = []
    centers = []
    scales = []

    for info in data:
        kpts.append(x['keypoints'])
        centers.append(x['pos'])
        scales.append(x['scale'])
    fp.close()

    return kpts, centers, scales

def generate_heatmap(heatmap, kpt, src_shape, gaussian_kernel):

    _, height, width = heatmap.shape
    for i in range(len(kpt)):
        if kpt[i][2] == 0:
            continue
        x = kpt[i][0]
        y = kpt[i][1]
        heatmap[i][int(1.0 * y * height / src_shape[0])][int(1.0 * x * width / src_shape[1])] = 1

    for i in range(len(kpt)):
        heatmap[i] = cv2.GaussianBlur(heatmap[i], gaussian_kernel, 0)
        am = np.amax(heatmap[i])
        heatmap[i] /= am / 255

    return heatmap

class CPNFolder(data.Dataset):

    def __init__(self, file_dir, output_shape, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], transformer=None):

        self.img_list = read_data_file(file_dir[0])
        self.kpt_list, self.center_list, self.scale_list = read_json_file(file_dir[1])
        self.transformer = transformer
        self.mean = mean
        self.std = std
        self.output_shape = output_shape

    def __getitem__(self, index):

        img_path = self.img_list[index]

        img = np.array(cv2.imread(img_path), dtype=np.float32)

        kpt = self.kpt_list[index]       # shape (num_points, 3)
        center = self.center_list[index] # shape (2)
        scale = self.scale_list[index]   # shape (1)

        img, kpt, center = self.transformer(img, kpt, center, scale)

        height, width, _ = img.shape

        label15 = np.zeros((len(kpt[0]), output_shape[0], output_shape[1]), dtype=np.float32)
        label15  = generate_heatmap(label15, kpt, (height, width), (15, 15))
        label11 = np.zeros((len(kpt[0]), output_shape[0], output_shape[1]), dtype=np.float32)
        label11  = generate_heatmap(label11, kpt, (height, width), (11, 11))
        label9 = np.zeros((len(kpt[0]), output_shape[0], output_shape[1]), dtype=np.float32)
        label9  = generate_heatmap(label9, kpt, (height, width), (9, 9))
        label7 = np.zeros((len(kpt[0]), output_shape[0], output_shape[1]), dtype=np.float32)
        label7  = generate_heatmap(label7, kpt, (height, width), (7, 7))
        valid = np.array(kpt[:, 2], dtype=float32)
        #label15[:,:,0] = 1.0 - np.max(label15[:,:,1:], axis=2) # for background

        img = img.transpose((2, 0, 1))
        img = Mytransforms.normalize(Mytransforms.to_tensor(img), self.mean, self.std)
        label15 = Mytransforms.normalize(Mytransforms.to_tensor(label15))
        label11 = Mytransforms.normalize(Mytransforms.to_tensor(label11))
        label9 = Mytransforms.normalize(Mytransforms.to_tensor(label9))
        label7 = Mytransforms.normalize(Mytransforms.to_tensor(label7))
        valid = Mytransforms.normalize(Mytransforms.to_tensor(valid))

        return img, label15, label11, label9, label7, valid 

    def __len__(self):

        return len(self.img_list)
