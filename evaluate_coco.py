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
from datasets.coco import COCODataset
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from utils.fileutils import get_files
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

data_dir = "/home/zjb/workspace/datasets/COCO/coco2017"
split = "val"
save_dir = "results/coco2017"
model_path = "saved_models/ctdet_coco_dla_2x.pth"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
img_dir = os.path.join(data_dir, 'images', '{}2017'.format(split))
annot_path = os.path.join(data_dir, 'annotations',
                          'instances_{}2017.json').format(split)

# model = ResNet(18, heads={'hm': COCODataset.num_classes, 'wh': 2, 'reg': 2})
model = DLANet(34, heads={'hm': COCODataset.num_classes, 'wh': 2, 'reg': 2}, pretrained=True)
device = torch.device('cuda')
model = torch.nn.DataParallel(model).cuda()
state_dict = torch.load(model_path)
print(state_dict['epoch'])
model.load_state_dict(state_dict['state_dict'])
model.eval()

coco_dataset = COCODataset(data_dir, split)
results = {}
for ind in tqdm(range(coco_dataset.num_samples)):
    img_id = coco_dataset.images[ind]
    img_info = coco_dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    # cv2.imshow(img_name, image)
    # cv2.waitKey()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean, std = coco_dataset.mean, coco_dataset.std
    res = process_results_nothr(model, image, mean, std, num_classes=coco_dataset.num_classes,  thresh=0.0)
    results[img_id] = res
coco_dataset.run_coco_eval(results, save_dir)

