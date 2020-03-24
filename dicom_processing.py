#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:01:48 2020

@author: wenzhao
"""
import pydicom as dicom
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
# from skimage import morphology
from skimage import measure
#from skimage.transform import resize
#from sklearn.cluster import KMeans
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#from plotly.tools import FigureFactory as FF
#from plotly.graph_objs import *

data_path = "data/S150058"
output_path = "data/images/"

def load_scan(files):
    # read file and sort according to their instance number 
    slices = [ dicom.read_file(s) for s in files ]
    slices.sort(key = lambda x: float(x.InstanceNumber))
    
    indexes = [ x.InstanceNumber for x in slices ]
    # convert to dictionary with key: instanceNumber, val: index 
    hash_map = {k: v for v, k in enumerate(indexes)}
    
    new_structure = []
    for i in range(1, 141): # instance number starts from 1
        one_scan = []
        for j in range(48):
            if (i + j * 140) in hash_map.keys():
                # one_scan.append(hash_map[i + j * 140])
                one_scan.append(slices[hash_map[i + j * 140]])
        new_structure.append(one_scan)
    '''
    # add this sliceThickness to the filfe
    for slice in new_structure:
        for element in slice:
            try:
                element.SliceThickness = np.abs(slice[1].ImagePositionPatient[2] - slice[0].ImagePositionPatient[2])
            except:
                element.SliceThickness = np.abs(slice[1].SliceLocation - slice[1].SliceLocation)       
    '''
    return new_structure

def get_pixels_hu(scans):
    # get all the iamge matrix from scans
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    for slice_number in range(len(scans)):
        # Convert to Hounsfield units (HU)
        intercept = scans[0].RescaleIntercept if 'RescaleIntercept' in scans[0] else -1024
        slope = scans[0].RescaleSlope if 'RescaleSlope' in scans[0] else 1
            
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
        
    return np.array(image, dtype=np.int16)

def sample_images(stack, rows=4, cols=4, start_with=0, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

def plot_HU(array_pixel):
    # plot HU dist graph
    plt.hist(first_slice_pixels.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

files = glob(data_path + '/*.dcm')
slices = load_scan(files)
first_slice = slices[0]
first_slice_pixels = get_pixels_hu(first_slice)


# plot the HUdistgraph
plot_HU(first_slice)
# grid plot the images
sample_images(first_slice_pixels)

print("Slice Thickness: %f" % first_slice[0].SliceThickness)
print("Pixel Spacing (row, col): (%f, %f) " % (first_slice[0].PixelSpacing[0], first_slice[0].PixelSpacing[1]))

'''
# save to local
id=1
np.save(output_path + "fullimages_%d.npy" % (id), first_slice_pixels)

# load from local
file_used=output_path+"fullimages_%d.npy" % id
imgs_to_process = np.load(file_used)[0].astype(np.float64) 
'''

"""
    VIZ
"""
# id = 1
# imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))
# grid images

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

pix_resampled, spacing = resample(first_slice_pixels, first_slice, np.array([1,1,1]))
print("Shape before resampling\t", first_slice_pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)

# VIZ static
def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.50)
    face_color = [115.2, 115.2, 192]
    mesh.set_facecolor([x/256 for x in face_color])
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.savefig('figure.png')

plot_3d(pix_resampled, 0)

# VIZ interactive
from plotly.offline import plot, iplot
from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go

def make_mesh(image, threshold=-300, step_size=1):

    print("Transposing surface")
    p = image.transpose(2,1,0)
    
    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold,step_size=step_size, allow_degenerate=True, use_classic=True) 
    return verts, faces

def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    print("Drawing")
    
    # Make the colormap single color since the axes are positional not intensity. 
#    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
      
    colormap=['rgb(115.2, 115.2, 192)','rgb(236, 236, 212)']
    trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker={'size': 1, 'opacity': 0.4})
    
    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    '''
    fig = FF.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization",
                        height=1000, 
                        width=1500,
                        aspectratio=dict(x=1, y=1, z=1)
                        )
    '''
    plot(fig)

v, f = make_mesh(pix_resampled, -300)
plotly_3d(v, f)
