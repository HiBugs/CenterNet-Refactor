#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu@zju.edu.cn
@Date: 2020/11/27
@Description:
"""
import numpy as np


def gaussian_csp(kernel):
    sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
    s = 2 * (sigma ** 2)
    # with random, not int(kernel / 2)?
    dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
    # if kernel % 2 == 0 & np.random.randint(0, 2) == 0:
    # 	dx = np.exp(-np.square(np.arange(kernel) - int((kernel - 1) / 2)) / s)
    # else:
    # 	dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
    return np.reshape(dx, (-1, 1))


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = _gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def _gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

