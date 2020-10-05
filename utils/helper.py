import os

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.nn import functional as F
import torch
from scipy.sparse import csgraph
from utils.data import load_fmri_data, list_2_tensor
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import random
import functools
import operator


def functools_reduce_iconcat(a):
    return functools.reduce(operator.iconcat, a, [])
def plot_confusion_matrix(label_truth, label_pred, save_path=None):
    label_pred = functools_reduce_iconcat(label_pred)
    label_truth = functools_reduce_iconcat(label_truth)
    cm = confusion_matrix(label_truth, label_pred)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative', 'Positive']
    plt.title('Versicolor or Not Versicolor Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    if save_path:
        plt.savefig(os.path.join(save_path, 'Confusion_matrix.png'))
    plt.show()


def train_loader(batch_size, input, target, mode='train'):
    assert len(input) == len(target), \
        "length of train_subject({}) should be the same as train_labels({})".format(
            len(input), len(target))

    # Define loaders
    # train_idx, valid_idx = train_test_split(np.arange(len(target)), test_size=0.2, shuffle=True, stratify=target)
    train_data, test_data, train_label, test_label = train_test_split(input, target, test_size=0.15, random_state=0)

    if mode == 'train':
        input_data = train_data  # convert to tensor
        label_data = train_label
    elif mode == 'test':
        input_data = test_data  # convert to tensor
        label_data = test_label

    subject_length = len(input_data)
    index_list = list(range(subject_length))

    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        subjects_list = []
        labels_list = []
        for i in index_list:
            subjects_list.append(input_data[i])
            labels_list.append(label_data[i])
            if len(subjects_list) == batch_size:
                yield list_2_tensor(subjects_list), list_2_tensor(labels_list)
                subjects_list = []
                labels_list = []

        # if the left sample is smaller than the batch size，
        # then the rest of the data form a mini-batch of len(subject_list)
        if len(subjects_list) > 0:
            yield list_2_tensor(subjects_list), list_2_tensor(labels_list)

    return data_generator


def train_vec_loader_2(batch_size, input, target, train_index, test_index, mode='train'):
    assert len(input) == len(target), \
        "length of train_subject({}) should be the same as train_labels({})".format(
            len(input), len(target))

    train_data = input[train_index]
    test_data = input[test_index]
    train_label = target[train_index]
    test_label = target[test_index]

    if mode == 'train':
        input_data = train_data  # convert to tensor
        label_data = train_label
    elif mode == 'test':
        input_data = test_data  # convert to tensor
        label_data = test_label

    subject_length = len(input_data)
    index_list = list(range(subject_length))

    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        subjects_list = []
        labels_list = []
        for i in index_list:
            subjects_list.append(input_data[i])
            labels_list.append(label_data[i])
            if len(subjects_list) == batch_size:
                yield list_2_tensor(subjects_list), list_2_tensor(labels_list)
                subjects_list = []
                labels_list = []

        # if the left sample is smaller than the batch size，
        # then the rest of the data form a mini-batch of len(subject_list)
        if len(subjects_list) > 0:
            yield list_2_tensor(subjects_list), list_2_tensor(labels_list)

    return data_generator


def cross_validation_train_vec_loader(input, cv):
    kfold = KFold(cv, True, random_state=0)
    return kfold.split(input)


def train_loader_graph(mode, input, target, feature=None, batchsize=32):
    # Batch size used when loading dat a
    assert len(input) == len(target), \
        "length of train_subject({}) should be the same as train_labels({})".format(
            len(input), len(target))

    # Define loaders
    # train_idx, valid_idx = train_test_split(np.arange(len(target)), test_size=0.2, shuffle=True, stratify=target)
    train_data, test_data, train_label, test_label, train_feat, test_feat = train_test_split(input, target, feature,
                                                                                             test_size=0.15,
                                                                                             random_state=0)
    if mode == 'train':
        input_data = train_data  # convert to tensor
        label_data = train_label
        feat_data = train_feat
    elif mode == 'test':
        input_data = test_data  # convert to tensor
        label_data = test_label
        feat_data = test_feat

    subject_length = len(input_data)
    index_list = list(range(subject_length))

    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        subjects_list = []
        labels_list = []
        feat_list = []
        for i in index_list:
            subjects_list.append(input_data[i])
            labels_list.append(label_data[i])
            feat_list.append(feat_data[i])
            if len(subjects_list) == batchsize:
                yield list_2_tensor(subjects_list), list_2_tensor(labels_list), list_2_tensor(feat_list)
                subjects_list = []
                labels_list = []
                feat_list = []

        # if the left sample is smaller than the batch size，
        # then the rest of the data form a mini-batch of len(subject_list)
        if len(subjects_list) > 0:
            yield list_2_tensor(subjects_list), list_2_tensor(labels_list), list_2_tensor(feat_list)

    return data_generator


def plot_train_result(history, best_epoch=None, save_path=None):
    """
        Display training and validation loss evolution over epochs
        Params
        -------
        history :
        best_epoch : int
            If early stopping is used, display the last saved model
        save_path : string
            Path to save the figure

        Return
        --------
        a matplotlib Figure
    """
    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])  # 1st row, entire row : global VAE loss
    ax2 = fig.add_subplot(gs[1, :])  # 2ndst row, entire row  on 2 * 2 grid: reconstruction

    # ax3 = fig.add_subplot(gs[2, 0])  # top left on a 4x4 grid: KL divergence
    # ax4 = fig.add_subplot(gs[2, 1])  # bottom right on a 4x4 grid: MI


    #  plot the overall loss
    ax1.set_title('Loss')
    ax1.plot(history['train_loss'], color='dodgerblue', label='train')
    ax1.plot(history['test_loss'], linestyle='--', color='lightsalmon', label='test')
    if best_epoch:
        ax1.axvline(best_epoch, linestyle='--', color='r', label='Early stopping')

    ax2.set_title('Accuracy')
    ax2.plot(history['train_acc'], color='dodgerblue', label='train')
    ax2.plot(history['test_acc'], linestyle='--', color='lightsalmon', label='test')

    ax1.legend()
    ax2.legend()

    if save_path:
        plt.savefig(os.path.join(save_path, 'loss_eval.png'))

    plt.show()

    return fig


def masked_loss(out, label, mask):
    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc


# def laplacian(mx, norm):
#     """Laplacian-normalize sparse matrix"""
#     assert (all (len(row) == len(mx) for row in mx)), "Input should be a square matrix"
#
#     return csgraph.laplacian(adj, normed = norm)

def num_correct(output, labels):
    """
        return the number correct
        -------
        :param
        output : tensor
            the output of model
        labels : tensor
            ground truth
        :return
        num of correct prediction : int
    """
    corr = len(labels)
    if output.shape[1] > 1:
        ### log softnax and Cross Entropy
        # if len(labels) != len(output):
        #     print(len(labels), len(output))
        #     return 0
        pred = output.max(dim=-1)[-1]
        corr = pred.eq(labels).sum().item()
    else:
        ### BCE
        pred = output > 0.5
        corr = (torch.squeeze(pred) == labels).sum()

    return corr


def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices()  # [2, 49216]
    v = x._values()  # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    out = out * (1. / (1 - rate))

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)
    return res
