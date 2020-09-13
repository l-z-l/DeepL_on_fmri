#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:26:20 2020

@author: wenzhao
"""
"""
      ___     __   ___  
     |_ _|   / /  / _ \ 
      | |   / /  | (_) |
     |___| /_/    \___/  
"""
from glob import glob
import os

from nilearn import plotting, image, datasets
from nilearn.image import resample_img, mean_img, concat_imgs
import numpy as np

data_path = "data/nii"
files = glob(data_path + '/*.nii')
files.pop(69)
first_file = files
'''
first_vol = image.load_img(files[0]) # shape of (64, 64, 48, 140)
# mask training set
haxby = datasets.fetch_haxby_simple()
haxby_files = haxby.func[0]
haxby_img = image.load_img(vol)
mean_haxby = mean_img(haxby_img)
'''


images = [image.load_img(x) for x in files]
# 4d (64, 64, 48, 139)
vol = concat_imgs([images], auto_resample=True)
# 69 is the faulty file 5112_S150058_I303069.dcm
# mean
mean_fmri = mean_img(vol)
"""
            _      
           (_)     
    __   __ _  ____
    \ \ / /| ||_  /
     \ V / | | / / 
      \_/  |_|/___|
"""          
# smooth the edges of the image
plotting.plot_stat_map(mean_fmri, threshold=3)
# pass to a smooth function
smooth_anat_img = image.smooth_img(mean_img, fwhm=3)
plotting.plot_roi(smooth_anat_img, threshold=None)

"""
                            _    
                           | |   
     _ __ ___    __ _  ___ | | __
    | '_ ` _ \  / _` |/ __|| |/ /
    | | | | | || (_| |\__ \|   < 
    |_| |_| |_| \__,_||___/|_|\_\                       
"""
from nilearn.input_data import NiftiMapsMasker, NiftiMasker, NiftiLabelsMasker
 
# define a default mask
masker = NiftiMasker(standardize=True, mask_strategy='epi',
                           memory="nilearn_cache", memory_level=5,
                           smoothing_fwhm=8) # smooth the mask

#masker_label = NiftiLabelsMasker(first_file, standardize=True)
#time_series = masker_label.fit_transform(first_file)
# train mask 
masker = masker.fit(first_file)
# plot the mask
# 
# plotting.plot_roi(masker.mask_img_, bg_img=  title="default mask")
# masker.generate_report().save_as_html("mask_report.html")
# fit the mask to the image
# default_mask = masker.mask_img_.get_data().astype(np.bool)
vol_masked = masker.transform(first_file)# .astype(np.float64)

print("Trended: mean %.2f, std %.2f" % (np.mean(vol_masked), np.std(vol_masked)))

# plot voxels 
"""
                                                 _          _       
       ___  ___   _ __  _ __   _ __ ___    __ _ | |_  _ __ (_)__  __
      / __|/ _ \ | '__|| '__| | '_ ` _ \  / _` || __|| '__|| |\ \/ /
     | (__| (_) || |   | |    | | | | | || (_| || |_ | |   | | >  < 
      \___|\___/ |_|   |_|    |_| |_| |_| \__,_| \__||_|   |_|/_/\_\
                                                                    
"""
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')

selected = np.random.randint(low=0, high=vol_masked.shape[1], size=39)
masked_fmri = vol_masked[:, selected ]

correlation_matrix = correlation_measure.fit_transform([masked_fmri])[0]
