#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu@zju.edu.cn
@Date: 2020/11/27
@Description:
"""
import cv2
import numpy as np
import random


def brightness(image, min=0.5, max=2.0):
    '''
    Randomly change the brightness of the input image.

    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_br = np.random.uniform(min, max)

    # To protect against overflow: Calculate a mask for all pixels
    # where adjustment of the brightness would exceed the maximum
    # brightness value and set the value to the maximum at those pixels.
    mask = hsv[:, :, 2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
    hsv[:, :, 2] = v_channel

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def random_pave(image, gts, igs, pave_size, lms=None, limit=8):
    img_height, img_width = image.shape[0:2]
    pave_h, pave_w = pave_size
    # paved_image = np.zeros((pave_h, pave_w, 3), dtype=image.dtype)
    paved_image = np.ones((pave_h, pave_w, 3), dtype=image.dtype) * np.mean(image, dtype=int)
    pave_x = int(np.random.randint(0, pave_w - img_width + 1))
    pave_y = int(np.random.randint(0, pave_h - img_height + 1))
    paved_image[pave_y:pave_y + img_height, pave_x:pave_x + img_width] = image
    # pave detections
    if len(igs) > 0:
        igs[:, 0:4:2] += pave_x
        igs[:, 1:4:2] += pave_y
        keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & \
                  ((igs[:, 3] - igs[:, 1]) >= 8)
        igs = igs[keep_inds]

    if len(gts) > 0:
        gts[:, 0:4:2] += pave_x
        gts[:, 1:4:2] += pave_y
        keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit)
        gts = gts[keep_inds]

    if lms is not None and len(lms) > 0:
        lms[:, :, 0] += pave_x
        lms[:, :, 1] += pave_y

    return paved_image, gts, igs, lms

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32),inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    # print(src_dir, dst_dir)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [_brightness, _contrast, _saturation]
    random.shuffle(functions)

    gs = _grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    _lighting(data_rng, image, 0.1, eig_val, eig_vec)


def _grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _lighting(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def _blend(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def _saturation(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    _blend(alpha, image, gs[:, :, None])


def _brightness(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def _contrast(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    _blend(alpha, image, gs_mean)


def _get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)
