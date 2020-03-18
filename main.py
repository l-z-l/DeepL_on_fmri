#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:26:20 2020

@author: wenzhao
"""
from nilearn import plotting, image
from nilearn.image import mean_img
import pydicom


file_name = "data/S150058/ADNI_002_S_0295_MR_Resting_State_fMRI_br_raw_20120511095055857_1930_S150058_I303069.dcm"
img = image.load_img(file_name)

# smoothe the edges of the image
smooth_anat_img = image.smooth_img(img, fwhm=3)
plotting.plot_roi(smooth_anat_img, threshold=None)


# masking 
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=file_name, standardize=True)

fmri_masked = masker.fit_transform(  )


import dcm2niix
