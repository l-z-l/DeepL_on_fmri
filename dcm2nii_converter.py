#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:58:45 2020

@author: wenzhao

https://nipype.readthedocs.io/en/0.12.1/interfaces/generated/nipype.interfaces.dcm2nii.html
"""

from nipype.interfaces.dcm2nii import Dcm2nii

converter = Dcm2nii()
# add list of files
converter.inputs.source_names = 
converter.inputs.gzip_output = True
converter.inputs.output_dir = './data/nii'