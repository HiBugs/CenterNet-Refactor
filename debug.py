#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu@zju.edu.cn
@Date: 2021/1/26
@Description:
"""
import cv2
import pdb
import numpy as np
from utils.img_augment import brightness
x = np.array([[[0.1],[0.5],[0.3]],
             [[0.4],[0.2],[0.6]]])

print(np.sum(x>0.2))
# img = brightness(img)
# img = cv2.flip(img, 1)
# inp = (img.astype(np.float32) / 255.)
# print(inp)
