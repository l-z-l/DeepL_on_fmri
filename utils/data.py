import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import networkx as nx
import bct
import torch
from nilearn.connectome import ConnectivityMeasure

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
        adj[adj != 0] = 1  # weighted graph
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


if __name__ == "__main__":
    # LOAD data
    ROI_signals, labels, labels_idex = load_fmri_data(dataset='273_MSDL')
    # ROI_signals[155] = np.nan_to_num(ROI_signals[155])
    # convert to functional connectivity
    connectivities = signal_to_connectivities(ROI_signals, kind='correlation', vectorize=True)
    # connectivities, _ = threshold(connectivities[:2])
    #
    # # inital node embeddings
    # H_0 = node_embed(connectivities)
    # H_0 = normalize_features(H_0)
    # # initial edge embeddings
    # W_0 = torch.as_tensor(connectivities)  # TODO: implement edge_embed() function
    # sparse_adj_list = sym_normalize_adj(connectivities)
