import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import networkx as nx
import os
import bct
import torch
from nilearn.connectome import ConnectivityMeasure
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def checkNone(matrix_list):
    '''
    TODO: ROI_signals[155] = np.nan_to_num(ROI_signals[155])
    Params :
    --------
        -
    Returns :
    --------
    '''

    return np.argwhere(np.isnan(ROI_signals))


def threshold(correlation_matrices, threshold=1):
    '''
    filter the functional connectivity based on connectivity strength
    Params :
    --------
        - correlation_matrices (m * n * n np.array) : the weighted adjacency matrices
        - threshold (float) : percentage to keep
    Returns :
    --------
        - (m * n * n np.array), with diagonal 0
        - list of nx.graph TODO: latter
        TODO: Zelun allow percentage argument latter
    '''
    # graph_list = []
    for i, matrix in enumerate(correlation_matrices):
        # node is not self connected
        np.fill_diagonal(matrix, 0)

        mean, std = np.mean(abs(matrix)), np.std(abs(matrix))

        # THRESHOLD: remove WHEN abs(connectivity) < mean + 1 * std
        matrix[abs(matrix) <= (mean + 1 * std)] = 0
        ### convert to nx.graph
        # g = nx.from_numpy_matrix(matrix)
        # graph_list.append(g)
        # relabels the nodes to match the  stocks names
        # G = nx.relabel_nodes(G, lambda x: atlas_labels[x])
        # print(f'{i}th graph has {g.number_of_nodes()} nodes, {g.number_of_edges() / 2} edges')

    return correlation_matrices, None


def node_embed(correlation_matrices, mask_coord='MSDL', hand_crafted=True):
    '''
    embed each node
    Params :
    --------
        - correlation_matrices (m * n * n np.array) : the weighted adjacency matrices
        - mask_coord (str) : the name of the mask
        - hand_crafted (bool) : using hand_engineered feature or vector embedding TODO: latter
    Returns :
        - (n * nROI * nFeat torch.tensor) : the initial node embeddings
    '''
    ### load coordinates of mask
    coordinate = torch.tensor(np.load(f'./data/{mask_coord}_coordinates.npy', allow_pickle=True), dtype=torch.float)

    ### node embeddings using graph local measures
    H = []
    for i, matrix in enumerate(correlation_matrices):
        graph_measure = {
            # 1 node degrees
            "degree": bct.degrees_und(matrix),
            # 2 node strength
            "node_strength": bct.strengths_und(matrix),
            # 3 participation coefficient
            # 4 betweenness centrality
            "betweenness_centrality": bct.betweenness_bin(matrix) / ((len(matrix) - 1) * (len(matrix) - 2)),
            # 5 K-coreness centrality
            "kcoreness_centrality": bct.kcoreness_centrality_bd(matrix)[0],
            # 6 subgraph centrality
            "subgraph_centrality": bct.subgraph_centrality(matrix),
            # 7 eigenvector centrality
            "eigenvector_centrality": bct.eigenvector_centrality_und(matrix),
            # 8 PageRank centrality
            "pagerank_centrality": bct.pagerank_centrality(matrix, d=0.85),
            # 9 diversity coefficient
            # 10 local efficiency
            "local_efficiency": bct.efficiency_bin(matrix, local=True)
        }

        vec = []
        for key, val in graph_measure.items():
            vec.append(val)
        # add coordinates of the ROIs
        H_i = torch.cat((torch.FloatTensor(vec).T, coordinate), axis=1)
        H.append(H_i)

    return list_2_tensor(H)


def edge_embed(dataDir='../data', dataset='271_AAL', connectivity=True, verbose=True):
    '''
    TODO: edge embedding using correlation, partial correslation and L2 distance
    Params :
    --------
        -
    Returns :
        -
    '''
    pass


def signal_to_connectivities(signals, kind='correlation', discard_diagonal=True, vectorize=False):
    '''
    extract functional connectivity from time series signals
    Params :
    --------
        - signals (np.array {um_subject, time_frame, num_ROI}) : the ROI signals
        - kind : “correlation”, “partial correlation”, “tangent”, “covariance”, “precision”
    Returns :
    --------
        - (np.ndarray {num_subjects, ROI, ROI}) : functional connectivity matrix
    '''
    # define a correlation measure
    correlation_measure = ConnectivityMeasure(kind=kind, discard_diagonal=discard_diagonal, vectorize=vectorize)
    # transform to connectivity matrices
    return correlation_measure.fit_transform(signals)


def list_2_tensor(list_matrix, axis=0):
    '''
    concatenate list element to form a ndTensor
    Params :
    --------
        - list_matrix : list of tensors (2d or 3d)
        - axis : the axis of concatenation/stack
    Returns :
    --------
        - tensor with the first dim as num N
    '''
    return torch.stack([x for x in list_matrix], dim=axis)


def load_fmri_data(dataDir='../data', dataset='271_AAL', label=None, verbose=False):
    '''
    Load the Saved 3D ROI signals
    Params :
    --------
        - dataDir (str) : the path of the data directory
        - dataset (str) : the name of the dataset
        - label (list str) : the labels needed TODO: latter
    Returns :
        - subjects_list (m * t * ROIs np.array) :  time series signal data (271, 140, 116)
        - label_list (np.array {num_subject}) : the data labels ['CN', 'MCI' ... ]
        - classes_idx (np.array {num_subject}) : the label encoded index of data labels [0, 3 ... ]
    '''
    subjects_list = np.load(dataDir + "/" + dataset + ".npy", allow_pickle=True)
    label_list = np.load(dataDir + "/" + dataset + "_label.npy", allow_pickle=True)

    ### only take the specified labels in the list
    if label != None:
        select_idx = [i for i, x in enumerate(label_list) if x == "CN" or x == "AD"]
        subjects_list = subjects_list[select_idx]
        label_list = label_list[select_idx]

    classes, classes_idx, classes_count = np.unique(label_list, return_inverse=True, return_counts=True)

    if verbose:
        # TODO: print the information
        print(classes)
        print(classes_count)

    return subjects_list, label_list, classes_idx


### not used rn
def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_features(mx_list):
    '''
    normalize adjacency matrix.
    Params :
    --------
        - adj (n * n torch.tensor) : containing only 0 or 1
    Returns :
        TODO: Zelun check formula plz, but not first priority
        - (n * nROI * nFeat torch.tensor) : the normalised with nodal degrees
    '''
    matrices = []
    for i, mx in enumerate(mx_list):
        rowsum = np.array(mx.sum(0))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx.T).T
        matrices.append(torch.as_tensor(mx, dtype=torch.float))
    return list_2_tensor(matrices)


def sym_normalize_adj(connectivity_matrices):
    '''
    Symmetrically normalize adjacency matrix.
    Params :
    --------
        - connectivity_matrices (n * nROI * nROI torch.tensor) : vontaining only 0 or 1
    Returns :
        - list (n * torch.sparse.FloatTensor(indices, values, shape))
    '''
    # TODO: Zelun read this thoroughly
    sparse_matrices = []
    for i, adj in enumerate(connectivity_matrices):
        # adj[adj != 0] = 1  # weighted graph
        adj += sp.eye(adj.shape[0])  # A^hat = A + I
        rowsum = np.array(np.count_nonzero(adj, axis=1))  # D = Nodal degrees

        adj = sp.coo_matrix(adj)
        d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5 116 * 1 tensor
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5 -> diagnol matrix
        adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)  # .tocsr() # D^-0.5AD^0.5
        sparse_matrices.append(sparse_mx_to_torch_sparse_tensor(adj))
    return sparse_matrices  # list_2_tensor(sparse_matrices)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    '''
    Convert a scipy sparse matrix to a torch sparse tensor.
    Params :
    --------
        - sparse_mx (n*n csc_matrix) :
    Returns :
        - torch.sparse.FloatTensor(indices, values, shape)
    '''
    # since the adj is symmetrical, can be either csc or csr
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # convert to coordinate, like to_sparse
    ### extract attributes from sparse_mx
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


### not used rn
def chebyshev_polynomials(adj, k):
    """
    Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation).
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = sym_normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


######################################
#
# DATA augment related functions
#
######################################

def augment_with_window(window_size, subjects_list, label_list, stride_size=1):
    '''
    sliding window method
    :param window_size: the actual window size
    :param subjects_list: np.load('/content/drive/My Drive/data/ADNI_denoised/179_AAL.npy', allow_pickle=True)
    :param label_list: np.load('/content/drive/My Drive/data/ADNI/128_AAL_label.npy', allow_pickle=True)
    :param stride_size: skip number of frames
    :return: new augmented subjects and labels
    '''
    new_subjects_slices = subjects_list[:, 0:(window_size), :]
    print(new_subjects_slices.shape)
    new_label_slices = label_list

    for i in range(1, 140 - window_size + 1, stride_size):
        new_subjects_slices = np.append(new_subjects_slices, subjects_list[:, i:(i + window_size), :], axis=0)
        new_label_slices = np.append(new_label_slices, label_list)

    return new_subjects_slices, new_label_slices


# ### Example usage
#
# stride = 5
# window = 100
# new_subjects_slices, new_label_slices = augment_with_window(window, subjects_de_more_list, label_de_more_list, stride)
# print("new subject_size is {}, stride size is {}, window size is {}, on denoised data".format(new_subjects_slices.shape, stride, 20))
# np.save("/content/drive/My Drive/Colab Notebooks/data/data_augument/271_100_{}_sliced_AAL.npy".format(stride), new_subjects_slices) #(15488, 20, 116)
# np.save("/content/drive/My Drive/Colab Notebooks/data/data_augument/271_100_{}_sliced_AAL_label.npy".format(stride), new_label_slices) #15488
# print("check if works np.load('/content/drive/My Drive/Colab Notebooks/data/data_augument/271_100_{}_sliced_AAL.npy', allow_pickle=True)".format(stride))


def interpolation_frames(oneside_window_size, subjects_list, stride_size, func, start_index):
    '''
    This is the helper function of @ref augment_with_selection
    :param oneside_window_size: 1/2 window actually, e.g. actual_window = 3, then window = 1, actual_window = 5, then window = 2
    :param subjects_list: np.load('/content/drive/My Drive/data/ADNI_denoised/179_AAL.npy', allow_pickle=True)
    :param label_list: np.load('/content/drive/My Drive/data/ADNI/128_AAL_label.npy', allow_pickle=True)
    :param stride_size: skip ever n number of frames
    :param func: function name: MIN, MAX, MEAN, SINGLE. when apply SINGLE, window must be 0.
    :param start_index: which stride index are we up to # as we can append the frames more easilly
    :return: when the start_index = i, the subjects list has been returned.
    '''
    new_subjects_list = subjects_list[:, start_index::stride_size, :]
    for s in range(len(subjects_list)):
        for i in range(oneside_window_size, stride_size, 140 - oneside_window_size):
            val = subjects_list[s, (i - oneside_window_size):(i + oneside_window_size + 1), :]
        if oneside_window_size < 1:
            new_val = val
        if func == "MAX":
            new_val = np.max(val, axis=0)  # (116,)
        if func == "MIN":
            new_val = np.min(val, axis=0)
        if func == "MEAN":
            new_val = np.mean(val, axis=0)
        new_subjects_list[s, i, :] = new_val  # after taking min/max/mean inside window
    return new_subjects_list


def augment_with_selection(oneside_window_size, subjects_list, label_list, stride_size=10, mask='AAL', func='MEAN',
                           save='', verbose=False):
    '''
    New methods, interpolation, which could cover along the time with a smaller size by
    select every stride_size number of frames, and do a function over the window frames
    to capture more features.
    :param oneside_window_size: 1/2 window actually, e.g. actual_window = 3, then window = 1, actual_window = 5, then window = 2, window=0, just extract the frame
    :param subjects_list: np.load('/content/drive/My Drive/data/ADNI_denoised/179_AAL.npy', allow_pickle=True)
    :param label_list: np.load('/content/drive/My Drive/data/ADNI/128_AAL_label.npy', allow_pickle=True)
    :param stride_size: skip number of frames
    :param func: function name: MIN, MAX, MEAN, SINGLE. when apply SINGLE, window must be 0.
    :return: append the subjects together along the axis 0
    '''
    new_subjects_list = interpolation_frames(oneside_window_size, subjects_list, stride_size, func, 0)
    new_label_list = label_list
    for j in range(1, stride_size):
        new_subjects_list = np.append(new_subjects_list,
                                      interpolation_frames(oneside_window_size, subjects_list, stride_size, func, j),
                                      axis=0)
        new_label_list = np.append(new_label_list, label_list)
        if verbose:
            print(new_subjects_list.shape)
            print(new_label_list.shape)

    if save:
        save_path = save + "{}_{}_{}_{}_{}".format(len(new_subjects_list), stride_size, func, oneside_window_size,
                                                         mask)
        np.save(save_path, new_subjects_list)
        np.save(save_path + "_label", new_label_list)

    return new_subjects_list, new_label_list


# ### Example usage
# stride = 10
# window = 0 #this window is 1/2 window actually, e.g. actual_window = 3, then window = 1, actual_window = 5, then window = 2
# func_name = "MAX" # MIN, MAX, MEAN, SINGLE when apply SINGLE, window must be 0,
# new_subjects_slices, new_label_slices = augment_with_selection(window, subjects_de_more_list, label_de_more_list, stride, func_name)
# print("new subject_size is {}, stride size is {}, window size is {}, on denoised data".format(new_subjects_slices.shape, stride, window))
# np.save("/content/drive/My Drive/Colab Notebooks/data/data_augument/271_every_{}_{}_sliced_AAL.npy".format(stride, func_name), new_subjects_slices)
# np.save("/content/drive/My Drive/Colab Notebooks/data/data_augument/271_every_{}_{}_sliced_AAL_label.npy".format(stride, func_name), new_label_slices)
# print("saved")
# #116 2710 14

def cluster_based_on_correlation(ROI_signals, mask_label, n_clusters):
    '''
    For a subject's N ROIs, we can have a N*N correlation matrix.
    Using this correlation matrix, we cluster that N items in M bins,
    so that we can say those items in one bins behaves similar.
    we aims to group rois for each subject to reduce the dimension.
    :param ROI_signals: taken from load_fmri_data
    :param mask_length: e.g. atlas_labels = atlas['labels']
    :return: a N_subjects*defined_N_clusters*N_timepoints
    '''
    connectivity_measure_func = ConnectivityMeasure(kind='correlation')
    connectivities_list = connectivity_measure_func.fit_transform(ROI_signals)
    clustered_roi_signals = []
    for i in range(len(connectivities_list)):
        corr = connectivities_list[i]
        # in case not regular correlation, made it symmetric
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)
        # to make sure both positive and negative information can be recorded
        dissimilarity = 1 - np.abs(corr)
        # use hierarchical/agglomerative clustering linkage
        hierarchy = linkage(squareform(dissimilarity), method='average')
        # labels are which clusters they have been assigned
        cluster_labels = fcluster(hierarchy, n_clusters, criterion='maxclust')
        # print(len(np.unique(cluster_labels)))

        cols = {"labels": cluster_labels, "rois": list(range(0, len(mask_label)))}
        df = pd.DataFrame(cols)
        # {1:[2,3],...} means cluster 1 is formed by ROI 2 and 3
        # or we can change list(range(0, len(mask_label)) to mask_labels it will print names of clusters
        result_cluster = df.groupby('labels')['rois'].apply(list).to_dict()
        ret = []
        for k in result_cluster.keys():
            roi_index_list = result_cluster[k]
            selected = ROI_signals[i, :, roi_index_list]
            ret.append(np.mean(selected, axis=0))
        clustered_roi_signals.append(ret)
    return np.array(clustered_roi_signals) #271*20*140
### Example usage
# device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
# print("Available computing device: ", device)
# ROI_signals, labels, labels_index = load_fmri_data(dataDir='/content/drive/My Drive/data/ADNI_denoised/', dataset='271_AAL')
# labels = [x if (x == "CN") else "CD" for x in labels]
# classes, labels_index, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
# print(ROI_signals.shape)
# print(labels_index)
# # AAL atlases
# atlas = datasets.fetch_atlas_aal()
# # Loading atlas image stored in 'maps'
# atlas_filename = atlas['maps']  # shape (91, 109, 91)
# # Loading atlas data stored in 'labels'
# atlas_labels = atlas['labels']
# print(cluster_based_on_correlation(ROI_signals, atlas_labels, 20))

if __name__ == "__main__":
    ### LOAD data
    ROI_signals, labels, labels_idex = load_fmri_data(dataset='271_AAL20')
    # new_subjects_list, new_label_list = augment_with_selection(0, ROI_signals, labels, stride_size=10, mask='MSDL', func="MAX", save='../data/interpolation/')

    ### generate augmented data using sliding window and save
    mask = "AAL20"
    save = f'../data/interpolation/{mask}/'
    if save and not os.path.isdir(save):
        os.mkdir(save)
    for func in ['MAX', 'MEAN']:
        for oneside_window_size in range(0, 2):
            augment_with_selection(oneside_window_size, ROI_signals, labels, stride_size=10, mask=mask, func=func,
                                   save=save)

    # ROI_signals[155] = np.nan_to_num(ROI_signals[155])
    ### convert to functional connectivity
    # connectivities = signal_to_connectivities(ROI_signals, kind='correlation', vectorize=False)
    # connectivities, _ = threshold(connectivities[:2])
    #
    # # inital node embeddings
    # H_0 = node_embed(connectivities)
    # H_0 = normalize_features(H_0)
    # # initial edge embeddings
    # W_0 = torch.as_tensor(connectivities)  # TODO: implement edge_embed() function
    # sparse_adj_list = sym_normalize_adj(connectivities)

    ###
    # new = cluster_based_on_correlation(ROI_signals, labels, 20)