import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import bct
from scipy.sparse import csgraph
from data import load_fmri_data
from sklearn.model_selection import train_test_splits
import random

def train_loader(mode='train', input, target):
    # Batch size used when loading data
    BATCHSIZE = 32

    assert len(input) == len(target), \
        "length of train_subject({}) should be the same as train_labels({})".format(
            len(input), len(target))

    # Define loaders
    train_data, test_data, train_label, test_label = train_test_split(input,
                                                                      target,
                                                                      test_size=0.2,
                                                                      random_state=42)

    print("train shape{} & {}".format(train_data.shape, train_label.shape))
    print("test shape{} & {}".format(test_data.shape, test_label.shape))

    if mode == 'train':
        input_data = train_data
        label_data = train_label
    elif mode == 'test':
        input_data = test_data
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
            if len(subjects_list) == BATCHSIZE:
                yield torch.tensor(subjects_list), torch.tensor(labels_list)
                subjects_list = []
                labels_list = []

        # if the left sample is smaller than the batch size，
        # then the rest of the data form a mini-batch of len(subject_list)
        if len(subjects_list) > 0:
            yield torch.tensor(subjects_list), torch.tensor(labels_list)

    return data_generator


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

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

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
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    out = out * (1./ (1-rate))

    return out

def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)
    return res
