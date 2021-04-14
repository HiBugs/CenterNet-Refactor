# -*- coding:utf-8 -*-

"""只需要按照实际改写images，annotations，categories另外两个字段其实可以忽略
在keypoints，categories内容是固定的不需修改
"""

import json
from tqdm import tqdm
import cv2
import os
import numpy as np
import re


class COCO(object):
    @staticmethod
    def info():
        return {"version": "1.0",
                "year": 2020,
                "contributor": "Mr.Zhu",
                "date_created": "2020/02/11",
                "github": "https://github.com/hibugs"}

    @staticmethod
    def licenses():
        return [{
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "name": "Attribution-NonCommercial-ShareAlike License",
            "id": 1}]

    @staticmethod
    def image():
        return {
            "file_name": "1.jpg",
            "height": 640,
            "width": 640,
            "id": 1     # 图片的ID编号（每张图片ID是唯一的）
        }

    @staticmethod
    def annotation():
        return {
            "area": 0.,     # 区域面积
            "image_id": 1,      # 对应的图片ID（与images中的ID对应）
            "bbox": [0., 0., 0., 0.],   # 定位边框 [x,y,w,h]
            "category_id": 1,   # 类别ID（与categories中的ID对应）
            "id": 10     # 对象ID，因为每一个图像有不止一个对象，所以要对每一个对象编号（每个对象的ID是唯一的）
            }

    @staticmethod
    def categories():
        return {
                "supercategory": "face",    # 主类别
                "id": 1,    # 类对应的id （0 默认为背景）
                "name": "face",     # 子类别
                }


class Wider2Coco(COCO):
    def __init__(self, txt_path, save_json_path, images_path):
        self.data = open(txt_path)
        self.save_json_path = save_json_path    # 最终保存的json文件
        self.images_path = images_path    # 原始图片保存的位置
        self.images = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.num = 1

    def __call__(self):
        print("Processing...")
        while True:
            img_path = self.data.readline()[:-1]
            if not img_path:
                break
            if re.search('jpg', img_path):
                if not os.path.exists(os.path.join(self.images_path, img_path)):
                    print(img_path, "NOT EXISTS.")
                    continue
                # init image
                image = self.image()
                image["file_name"] = img_path
                image["id"] = self.num
                img = cv2.imread(os.path.join(self.images_path, img_path))
                if img is None:
                    print(img_path, "Image Error.")
                    continue
                image["height"] = img.shape[0]
                image["width"] = img.shape[1]

                line = self.data.readline()[:-1]
                if not line:
                    break
                face_num = int(line)
                # init annotation
                annotation = self.annotation()
                for j in range(face_num):
                    line = [float(x) for x in self.data.readline().strip().split()]
                    bbox = list(line[:4])
                    annotation["image_id"] = self.num
                    annotation["id"] = self.annID
                    annotation["bbox"] = bbox
                    annotation['area'] = bbox[2]*bbox[3]
                    self.annotations.append(annotation)

                    self.annID += 1     # 对应对象
                    annotation = self.annotation()

                self.num += 1       # 对应图像
                self.images.append(image)

        jsdata = {"info": self.info(), "licenses": self.licenses(), "images": self.images,
                  "annotations": self.annotations, "categories": [self.categories()]}
        json.dump(jsdata, open(self.save_json_path, 'w'), indent=4, default=float)      # python3 需加上default=float 否则会报错


if __name__ == '__main__':
    img_path = '/home/zjb/workspace/datasets/WiderFace/WIDER_train/images'
    txt_path = '/home/zjb/workspace/datasets/WiderFace/wider_face_split/wider_face_train_bbx_gt.txt'
    save_path = '/home/zjb/workspace/datasets/WiderFace/WIDER_train/train.json'

    Wider2Coco(txt_path, save_path, img_path)()
    Wider2Coco(txt_path.replace("train", "val"),
               save_path.replace("train", "val"),
               img_path.replace("train", "val"))()
