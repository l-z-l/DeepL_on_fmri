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

# specify the file directory
data_path = "data/S150058"
files = glob(data_path + '/*.dcm')

# read file and sort according to their instance number 
slices = [ dicom.read_file(s) for s in files ]
slices.sort(key = lambda x: float(x.InstanceNumber))
indexes = [ x.InstanceNumber for x in slices ]



# convert to dictionary with key: instanceNumber, val: index 
hash_map = {k: v for v, k in enumerate(indexes)}

# show the position
# position = [x.ImagePositionPatient[2] for x in slices]

new_structure = []
for i in range(1, 141): # instance number starts from 1
    one_scan = []
    for j in range(48):
        if (i + j * 140) in hash_map.keys():
            one_scan.append(hash_map[i + j * 140])
            # one_scan.append(slices[hash_map[i + j * 140]])
    new_structure.append(one_scan)
            
'''
# 4976 5112

sort_index = np.argsort(slices)
sorted_array = np.sort(slices)

frames = np.ceil(len(slices) / 140)

#for i in range(140):
 #   for j in range(frames):
'''