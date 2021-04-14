from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("..")

import os
import cv2
import math
import json
import random
import numpy as np
from utils.gaussian import gaussian_radius, draw_umich_gaussian, gaussian_csp
from utils.img_augment import affine_transform, get_affine_transform, color_aug

import pycocotools.coco as coco
import torch.utils.data as data
from pycocotools.cocoeval import COCOeval


class COCODataset(data.Dataset):
    num_classes = 80
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    class_name = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, data_dir, split):
        super(COCODataset, self).__init__()
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, 'images', '{}2017'.format(split))
        self.annot_path = os.path.join(
            self.data_dir, 'annotations',
            'instances_{}2017.json').format(split)

        self.max_objs = 128

        self._valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split

        print('==> initializing coco 2017 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

        self.keep_res = False

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        # print(file_name)
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  # 中心点

        if self.keep_res:
            input_h = (height | 31) + 1
            input_w = (width | 31) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.default_resolution

        flipped = False
        if self.split == 'train':
            # sf = 0.4
            # cf = 0.1
            # c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            # c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            # s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            # if np.random.randint(0, 2) == 0:
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, img.shape[1])
            h_border = self._get_border(128, img.shape[0])
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

            if np.random.randint(0, 2) == 0:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train':
            if np.random.randint(0, 2) == 0:
                color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        # cv2.imshow("temp", inp)
        # cv2.waitKey()

        # 归一化
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        down_ratio = 4
        output_h = input_h // down_ratio
        output_w = input_w // down_ratio
        # num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        # temp = 0
        for k in range(num_objs):  # num_objs图中标记物数目
            ann = anns[k]  # 第几个标记物的标签
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)  # 将box坐标转换到 128*128 内的坐标
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

            # 上面几行都是做数据扩充和resize之后的变换，不重要
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                # x1, y1, x2, y2 = int(np.ceil(bbox[0])), int(np.ceil(bbox[1])), int(bbox[2]), int(bbox[3])
                # c_x, c_y = int(bbox[0] + bbox[2]) // 2, int(bbox[1] + bbox[3]) // 2
                # c_x = max(1, c_x)
                # c_x = min(hm.shape[2] - 2, c_x)
                # c_y = max(1, c_y)
                # c_y = min(hm.shape[1] - 2, c_y)
                # dx = gaussian_csp(x2 - x1)
                # dy = gaussian_csp(y2 - y1)
                # gau_map = np.multiply(dy, np.transpose(dx))
                # hm[cls_id, y1:y2, x1:x2] = np.maximum(hm[cls_id, y1:y2, x1:x2], gau_map)
                # hm[cls_id, c_y, c_x] = 1
                #
                # wh[k] = np.log(w), np.log(h)
                # ind[k] = output_w * c_y + c_x
                # reg[k] = (bbox[0] + bbox[2]) / 2 - c_x - 0.5, (bbox[1] + bbox[3]) / 2 - c_y - 0.5
                # reg_mask[k] = 1

                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                # wh[k] = np.log(w), np.log(h)
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
            # else:
            #     temp += 1
        # print(num_objs, temp)
        # print(hm.shape)
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        reg_offset_flag = True
        if reg_offset_flag:
            ret.update({'reg': reg})
        return ret

    def __len__(self):
        return self.num_samples

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

    @staticmethod
    def _get_border(border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    @staticmethod
    def _coco_box_to_bbox(box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    @staticmethod
    def _to_float(x):
        return float("{:.2f}".format(x))


if __name__ == '__main__':
    import torch
    train_dataset = COCODataset(data_dir='../../datasets/COCO/coco2017', split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    for i, batch in enumerate(train_loader):
        # data, label = batch[0], batch[1]
        # print(data.size())
        # print(label)
        break