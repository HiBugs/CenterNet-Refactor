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
from tqdm import tqdm
from networks.resnet import ResNet
from networks.dla_dcn import DLANet
from networks.post_process_new import process_results_nothr, process_results
from datasets.face_dataset import FaceDataset
from utils.fileutils import get_files
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

img_dir = "/home/zjb/workspace/datasets/WiderFace/WIDER_val/images"
save_dir = "results/widerface"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = ResNet(18)
# model = DLANet(34)
device = torch.device('cuda')
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('saved_models/WiderFace/best_resnet18.pth'))
model.eval()

img_list = get_files(img_dir)
for img_name in tqdm(img_list):
    image = cv2.imread(img_name)
    # cv2.imshow(img_name, image)
    # cv2.waitKey()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean, std = FaceDataset.mean, FaceDataset.std
    res = process_results_nothr(model, image, mean, std)
    # print(res)
    # print(len(res))
    # print(res[1])
    # import pdb; pdb.set_trace()

    FaceDataset.save_widerface_results(res, img_name, save_dir)

"""
WiderFace-Evaluation$ python evaluation.py -p ../CenterNet_pytorch/results/widerface/ -g ../datasets/WiderFace/wider_face_split/
"""