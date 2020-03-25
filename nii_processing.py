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
'''
first_vol = image.load_img(files[0]) # shape of (64, 64, 48, 140)
# mask training set
haxby = datasets.fetch_haxby_simple()
haxby_files = haxby.func[0]
haxby_img = image.load_img(vol)
mean_haxby = mean_img(haxby_img)
'''
images = [image.load_img(x) for x in files]
images.pop(69)
# 4d (64, 64, 48, 139)
vol = concat_imgs([images[:69],images[70:]], auto_resample=True)
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
masker = NiftiMasker(standardize=True, mask_strategy='background', high_pass = 0.8,
                           memory="nilearn_cache", memory_level=2,
                           smoothing_fwhm=8) # smooth the mask
# train mask 
masker = masker.fit(images)
# plot the mask
plotting.plot_roi(masker.mask_img_, bg_img=mean_fmri, title="default mask")
masker.generate_report().save_as_html("mask_report.html")
# fit the mask to the image
# default_mask = masker.mask_img_.get_data().astype(np.bool)
vol_masked = masker.fit_transform(images)# .astype(np.float64)

print("Trended: mean %.2f, std %.2f" % (np.mean(vol_masked), np.std(vol_masked)))
"""
                                             _          _       
   ___  ___   _ __  _ __   _ __ ___    __ _ | |_  _ __ (_)__  __
  / __|/ _ \ | '__|| '__| | '_ ` _ \  / _` || __|| '__|| |\ \/ /
 | (__| (_) || |   | |    | | | | | || (_| || |_ | |   | | >  < 
  \___|\___/ |_|   |_|    |_| |_| |_| \__,_| \__||_|   |_|/_/\_\
                                                                
"""

from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
# correlation_matrix = correlation_measure.fit_transform([time_series])[0]
