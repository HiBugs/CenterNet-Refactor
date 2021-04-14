#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu@zju.edu.cn
@Date: 2020/12/10
@Description:
"""
import os


def get_files(path, suffix='.jpg'):
    file_list = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1] == suffix:
                file_list.append(os.path.join(root, f))

    return file_list
