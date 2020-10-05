import os.path
import random

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset

from utils.data import load_fmri_data, signal_to_connectivities, node_embed
import numpy as np
import scipy.sparse as sp
import networkx as nx
from torch_geometric.data import Data, DataLoader

# class GraphDataset(Dataset):
#     def __init__(self, root_file, transform=None, pre_transform=None):
#         super(GraphDataset, self).__init__(root_file, transform, pre_transform)
#         self.root_file = root_file
#
#         ROIs, labels, labels_index = load_fmri_data(dataDir='data', dataset=root_file)
#         connectivity_matrices = signal_to_connectivities(ROIs, kind='correlation')
#
#         H_0 = node_embed(connectivity_matrices, 'Havard_Oxford')
#
#
#     @property
#     def raw_file_name(self):
#         return self.root_file
#
#     @property
#     def processed_file_names(self):
#         return self.root_file
#
#     def process(self):
#         # construct
#         pass
#
#     def len(self):
#         return len(self.processed_file_names)
#
#     def get(self, idx):
#         data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
#         return data
if __name__ == '__main__':
    # dataset = GraphDataset("273_MSDL.npy")
    ROIs, labels, labels_index = load_fmri_data(dataDir='../data', dataset='273_MSDL')
    connectivity_matrices = signal_to_connectivities(ROIs, kind='correlation')

    partial_corr = signal_to_connectivities(ROIs, kind='partial correlation')
    covariance = signal_to_connectivities(ROIs, kind='covariance')

    ### get features
    graphs = []
    for i, matrix in enumerate(connectivity_matrices):
        # node is not self connected
        np.fill_diagonal(matrix, 0)
        mean, std = np.mean(abs(matrix)), np.std(abs(matrix))

        ### THRESHOLD: remove WHEN abs(connectivity) < mean + 1 * std
        mask = abs(matrix) <= (mean + 0.5 * std)
        matrix[mask] = 0
        partial_corr[i][mask] = 0
        covariance[i][mask] = 0

        ### node_embed
        x = node_embed([matrix], mask_coord='MSDL').squeeze()

        ### edge_attr
        corr = sp.coo_matrix(matrix)
        par_corr = sp.coo_matrix(partial_corr[i])
        covar = sp.coo_matrix(covariance[i])
        edge_attr = torch.from_numpy(np.vstack((corr.data, par_corr.data, covar.data)).transpose())

        # convert to 0 or 1
        matrix[matrix != 0] = 1
        # np.fill_diagonal(matrix, 1)

        ### reshape edge tensor
        edge_index = sp.coo_matrix(matrix)
        edge_index = torch.from_numpy(np.vstack((edge_index.row, edge_index.col))).long()

        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels_index[i]))

        print(f"Data: {graphs[-1]}")
        # print(f'Is directed: {graphs[-1].is_undirected()}')
        print(f'Contains isolated nodes: {graphs[-1].contains_isolated_nodes()}')
        print(f'Self Connected: {graphs[-1].contains_self_loops()}')

    ### conver to sparse matrix
    random.shuffle(graphs)
    train_idx, valid_idx = train_test_split(np.arange(len(graphs)),
                                            test_size=0.15,
                                            shuffle=True)

    train_loader = DataLoader(train_idx, batch_size=32, shuffle=True)
    test_loader = DataLoader(valid_idx, batch_size=32)
