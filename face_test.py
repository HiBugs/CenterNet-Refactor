#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu@zju.edu.cn
@Date: 2020/11/27
@Description:
"""
import os
import cv2
import time
import torch
import numpy as np
from networks.resnet import ResNet
from networks.dla_dcn import DLANet
from networks.post_process_new import process_results
from datasets.face_dataset import FaceDataset
from datasets.coco import COCODataset
from utils.drawer import draw
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# def draw(filename, res):
#     img = cv2.imread(filename)
#     for class_name, lx, ly, rx, ry, prob in res:
#         # cv2.putText(img, "{:.2f}".format(prob), (int(lx), int(ly)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), thickness=2)
#         cv2.rectangle(img, (int(lx), int(ly)), (int(rx), int(ry)), color=(0, 255, 0), thickness=2)
#     cv2.imwrite(os.path.join('results', 'resnet34', os.path.split(filename)[-1]), img)
#     cv2.imshow("face", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    # model = ResNet(18, heads={'hm': COCODataset.num_classes, 'wh': 2, 'reg': 2}, pretrained=False)
    model = DLANet(34, heads={'hm': COCODataset.num_classes, 'wh': 2, 'reg': 2})
    device = torch.device('cuda')
    model = torch.nn.DataParallel(model).cuda()
    # model.load_state_dict(torch.load('saved_models/WiderFace/best_resnet18.pth'))
    model.load_state_dict(torch.load('saved_models/COCO2017/coco2017_best_dla34.pth'))
    model.eval()

    # for image_name in [os.path.join('images', f) for f in os.listdir('images')]:
    image_name = 'images/33823288584_1d21cf0a26_k.jpg'
    if image_name.split('.')[-1] == 'jpg':
        image = cv2.imread(image_name)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mean, std = FaceDataset.mean, FaceDataset.std
        mean, std = COCODataset.mean, COCODataset.std
        res = process_results(model, image, mean, std, threshold=0.1)
        draw(image, res, names=COCODataset.class_name)
