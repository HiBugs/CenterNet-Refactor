#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu@zju.edu.cn
@Date: 2020/12/10
@Description:
"""
import os
import cv2
import math
from PIL import Image, ImageDraw
import time
import torch
import numpy as np
import torch.nn as nn
from networks.resnet import ResNet
from networks.losses import gather_feat, transpose_and_gather_feat
from utils.img_augment import get_affine_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_results(model, image, mean, std, threshold=0.3):
    ret = process_results_nothr(model, image, mean, std)

    res = np.empty([1, 6])
    for i, c in ret.items():
        tmp_s = ret[i][ret[i][:, 4] > threshold]
        tmp_c = np.ones(len(tmp_s)) * (i + 1)
        tmp = np.c_[tmp_c, tmp_s]
        res = np.append(res, tmp, axis=0)
    res = np.delete(res, 0, 0)
    res = res.tolist()

    return res


def process_results_nothr(model, image, mean, std):
    images, meta = pre_process(image, mean, std)
    images = images.to(device)
    output, dets = process(model, images)
    dets = post_process(dets, meta)
    ret = merge_outputs(dets)

    return ret


def pre_process(image, mean, std):
    height, width = image.shape[0:2]
    # inp_height, inp_width = int(height * scale), int(width * scale)
    inp_height = (height | 31) + 1
    inp_width = (width | 31) + 1
    # inp_height, inp_width = int(inp_height*scale), int(inp_width*scale)
    # inp_height, inp_width = 1024, 1024
    # print(height, width)
    # print(scale, inp_height, inp_width)
    s = np.array([inp_width, inp_height], dtype=np.float32)
    c = np.array([width/2.,  height/2.], dtype=np.float32)
    # s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
    # cv2.imwrite("temp.jpg", inp_image)

    # mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
    # std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)

    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)  # 三维reshape到4维，（1，3，512，512）

    images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4}
    return images, meta


def process(model, images):
    with torch.no_grad():
        output = model(images)
        hm = output['hm'].sigmoid_()
        # ang = output['ang'].relu_()
        # wh = output['wh']
        wh = torch.exp(output['wh'])
        reg = output['reg']
        # torch.cuda.synchronize()
        dets = ctdet_decode(hm, wh, reg=reg, K=100)  # K 是最多保留几个目标
    return output, dets


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, K=100):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    reg = transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    # xs = xs.view(batch, K, 1)
    # ys = ys.view(batch, K, 1)
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)

    # ang = transpose_and_gather_feat(ang, inds)
    # ang = ang.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


def post_process(dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    num_classes = 1
    # dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], num_classes)
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        # dets[0][j][:, :5] /= 1
    return dets[0]


def merge_outputs(detections):
    num_classes = 1
    max_obj_per_img = 1000
    scores = np.hstack([detections[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_obj_per_img:
        kth = len(scores) - max_obj_per_img
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, 2 + 1):
            keep_inds = (detections[j][:, 4] >= thresh)
            detections[j] = detections[j][keep_inds]
    return detections

