# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import PIL.Image as Image

_im_zfile = []


def imread(filename, flags=cv2.IMREAD_COLOR):
    global _im_zfile
    path, img_name = os.path.split(filename)
    path_data, img_set = os.path.split(path)

    path_zip = os.path.join(path_data, img_set) + '.zip'
    path_img = os.path.join(img_set, img_name)
    if not os.path.isfile(path_zip):
        print("zip file '%s' is not found"%(path_zip))
        assert 0
    for i in range(len(_im_zfile)):
        if _im_zfile[i]['path'] == path_zip:
            data = _im_zfile[i]['zipfile'].read(path_img)
            return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

    _im_zfile.append({
        'path': path_zip,
        'zipfile': zipfile.ZipFile(path_zip, 'r')
    })
    data = _im_zfile[-1]['zipfile'].read(path_img)

    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)
