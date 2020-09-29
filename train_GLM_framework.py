# system
from glob import glob
import os
import json
from tqdm import tqdm

# data/img processing
import numpy as np
import pandas as pd
from skimage.transform import resize

# nilearn
import nibabel as nib
from nilearn import plotting
from nilearn.image import resample_img, load_img, mean_img, concat_imgs, index_img, iter_img, resample_to_img
from nilearn.input_data import NiftiMapsMasker, NiftiMasker
from nilearn import datasets
from nilearn.masking import apply_mask
from nilearn.connectome import ConnectivityMeasure
from nilearn.connectome import sym_matrix_to_vec

# ml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score, LeaveOneOut, KFold, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from utils.data import *
from models.GNN import GCN
from utils.config import args
from utils.helper import masked_loss, masked_acc
##########################################################
# %% Load Data
##########################################################
# LOAD data
ROIs, labels, labels_index = load_fmri_data(dataDir='data/', dataset='273_Havard_Oxford')
# convert to functional connectivity
connectivity_matrices = signal_to_connectivities(ROIs, kind='correlation', discard_diagonal=True, vectorize=True)

labels = [x if (x == "CN") else "CD" for x in labels]
classes, labels_index, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
label = torch.as_tensor(labels_index, dtype=torch.float)

model = LinearSVC()


# print(labels_index)
cv = StratifiedShuffleSplit(n_splits=5, random_state=42, test_size=0.2)

scores = []
print("Stratified Shuffle Split scores")
for train, test in cv.split(connectivity_matrices, labels_index):
    # *ConnectivityMeasure* can output the estimated subjects coefficients
    classifier = LinearSVC().fit(connectivity_matrices[train], labels_index[train])
    # make predictions for the left-out test subjects
    predictions = classifier.predict(connectivity_matrices[test])
    # store the accuracy for this cross-validation fold
    scores.append(accuracy_score(labels_index[test], predictions))
    # print(scores[-1])

print(np.mean(scores))
# trainX, testX, trainY, testY = train_test_split(connectivity_matrices, labels_index, test_size=0.15)
# model = model.fit(trainX, trainY)
# predictions = model.predict(testX).round()
# print(accuracy_score(testY, predictions))