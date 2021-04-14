#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu@zju.edu.cn
@Date: 2020/11/27
@Description:
"""
import sys
sys.path.append("..")

import os
import cv2
import math
import json
import random
import numpy as np
from utils.gaussian import gaussian_radius, draw_umich_gaussian
from utils.img_augment import affine_transform, get_affine_transform
import torch.utils.data as data
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval


class AirDataset(data.Dataset):
    num_classes = 1
    default_resolution = [512, 512]
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
    # mean = np.array([0.5194416012442385, 0.5378052387430711, 0.533462090585746], dtype=np.float32).reshape(1, 1, 3)
    # std = np.array([0.3001546018824507, 0.28620901391179554, 0.3014112676161966], dtype=np.float32).reshape(1, 1, 3)
    keep_res = False

    def __init__(self, data_dir='data', split='train'):
        # self.data_dir = os.path.join(data_dir, 'airplane')
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, 'images')
        try:
            if split == 'train':
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'train.json')
            elif split == 'val':
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'val.json')
        except:
            print('No any data!')

        self.max_objs = 128
        self.class_name = ['airplane']
        self._valid_ids = [i for i in range(1, self.num_classes+1)]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)

        self.split = split
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  # 中心点

        if self.keep_res:
            input_h = (height | 31) + 1
            input_w = (width | 31) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.default_resolution

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)

        # 归一化
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        down_ratio = 4
        output_h = input_h // down_ratio
        output_w = input_w // down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        ang = np.zeros((self.max_objs, 1), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        for k in range(num_objs):       # num_objs图中标记物数目
            ann = anns[k]       # 第几个标记物的标签
            bbox, an = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            bbox[:2] = affine_transform(bbox[:2], trans_output)     # 将box坐标转换到 128*128 内的坐标
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

            # 上面几行都是做数据扩充和resize之后的变换，不重要
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ang[k] = 1. * an
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'ang': ang}
        reg_offset_flag = True
        if reg_offset_flag:
            ret.update({'reg': reg})
        return ret

    @staticmethod
    def _coco_box_to_bbox(box):
        bbox = np.array([box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2],
                        dtype=np.float32)
        ang = float(box[4])
        return bbox, ang

    @staticmethod
    def _to_float(x):
        return float("{:.2f}".format(x))

    def _convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        flag = True
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))
                    if flag:
                        flag = False
                        print("image_id", image_id)
                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def _save_results(self, results, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        json.dump(self._convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_coco_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self._save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
