#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:33:54 2020

@author: wenzhao
"""
# reshape
import pydicom as dicom
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

data_path = "data/S150058"

files = glob(data_path + '/*.dcm')


slices = [dicom.read_file(s) for s in files]
slices.sort(key = lambda x: float(x.InstanceNumber))

index = [x.InstanceNumber for x in slices]
position = [x.ImagePositionPatient[2] for x in slices]

# 4976 5112

sort_index = np.argsort(slices)
sorted_array = np.sort(slices)

frames = np.ceil(len(slices) / 140)

#for i in range(140):
 #   for j in range(frames):