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

def threshold(correlation_matrices, threshold=1):
    '''
    filter the functional connectivity with percentage
    Params :
    --------
        - correlation_matrices (m * n * n np.array) : the weighted adjacency matrices
        - threshold (float) : percentage
    Returns :
    --------
        - adjacency matrix in n * n format, with diagonal 0
        - list of nx.graph
        TODO: allow percentage argument
    '''
    # Creates graph using the data of the correlation matrix
    # graph_list = []
    for i, matrix in enumerate(correlation_matrices):
        # node is not self connected
        np.fill_diagonal(matrix, 0)

        mean, std = np.mean(abs(matrix)), np.std(abs(matrix))

        # THRESHOLD: remove WHEN abs(connectivity) < mean + 1.5 * std
        # reduced from over 6000 to ~= 600 around 13.36%, since the data is normalised
        matrix[abs(matrix) <= (mean + 1 * std)] = 0
        ### convert to nx.graph
        # g = nx.from_numpy_matrix(matrix)
        # graph_list.append(g)
        # relabels the nodes to match the  stocks names
        # G = nx.relabel_nodes(G, lambda x: atlas_labels[x])
        # print(f'{i}th graph has {g.number_of_nodes()} nodes, {g.number_of_edges() / 2} edges')

    return correlation_matrices, None


def node_embed(correlation_matrices, mask_name='AAL', hand_crafted=True, dataDir='../data'):
    '''
    embed each node
    Params :
    --------
        - correlation_matrices (m * n * n np.array) : the weighted adjacency matrices
        - hand_crafted (bool) : using hand_engineered feature or vector embedding
    Returns :
        - adjacency matrix in (n_subjects, ) format, with diagonal 0
        TODO: Documentation + allow node2Vec embed
        :param dataDir:
    '''
    coordinate = torch.tensor(np.load(dataDir + '/' + mask_name + "_coordinates.npy", allow_pickle=True), dtype=torch.float)
    print(coordinate.shape)

    H = []
    for i, matrix in enumerate(correlation_matrices):
        # node embeddings using graph local measures
        graph_measure = {
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

def list_2_tensor(list_matrix):
    '''
    extract connectivities from time series signals
    Params :
    --------
        - list_tensors : list of tenosrs (2d or 3d)
    Returns :
    --------
        - tensor with dim 0 as num
    '''
    return torch.stack([x for x in list_matrix], dim=0)

def load_fmri_data(dataDir='../data', dataset='271_AAL', label=None, verbose=False):
    '''
    Load the Saved 3D ROI signals
    Params :
    --------
        - dataDir (str) : the path of the data directory
        - dataset (str) : the name of the dataset
        - label (list str) : the labels needed TODO: latter
    Returns :
        - subjects_list (m*t*ROIs np.array) :  time series signal data (271, 140, 116)
        - label_list (np.array {num_subject}) : the data labels ['CN', 'MCI' ... ]
        - classes_idx (np.array {num_subject}) : the label encoded index of data labels [0, 3 ... ]
    '''
    subjects_list = np.load(dataDir + "/" + dataset + ".npy", allow_pickle=True)
    label_list = np.load(dataDir + "/" + dataset + "_label.npy", allow_pickle=True)

    ### only take the specified labels in the list
    if label != None:
        select_idx = [i for i, x in enumerate(label_list) if x == "CN" or x =="AD"]
        subjects_list = subjects_list[select_idx]
        label_list = label_list[select_idx]

    classes, classes_idx, classes_count = np.unique(label_list, return_inverse=True, return_counts=True)

    if verbose:
        # TODO: print the information
        print(classes)
        print(classes_count)

    return subjects_list, label_list, classes_idx

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
    Symmetrically normalize adjacency matrix.
    Params :
    --------
        - adj (n * n torch.tensor) : vontaining only 0 or 1
    Returns :
        TODO: Zelun check formula
    '''
    matrices = []
    for i, mx in enumerate(mx_list):
        # mx = sp.coo_matrix(mx)
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        matrices.append(torch.as_tensor(mx))
    return list_2_tensor(matrices)

def sym_normalize_adj(connectivity_matrices):
    '''
    Symmetrically normalize adjacency matrix.
    Params :
    --------
        - adj (n * n torch.tensor) : vontaining only 0 or 1
    Returns :
        - list (n * torch.sparse.FloatTensor(indices, values, shape))
    '''
    sparse_matrices = []
    for i, adj in enumerate(connectivity_matrices):

        adj[adj != 0] = 1 # weighted graph
        adj += sp.eye(adj.shape[0]) # A^hat = A A+ I

        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))  # D
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
        adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt) #.tocsr() # D^-0.5AD^0.5
        sparse_matrices.append(sparse_mx_to_torch_sparse_tensor(adj))
    return sparse_matrices # list_2_tensor(sparse_matrices)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    '''
    Convert a scipy sparse matrix to a torch sparse tensor.
    Params :
    --------
        - sparse_mx (csc matrix) :
    Returns :
        - torch.sparse.FloatTensor(indices, values, shape)
    '''
    sparse_mx = sparse_mx.tocoo().astype(np.float32) # convert to coordinate, like to_sparse
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

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
    ROI_signals, labels, labels_idex = load_fmri_data()
    # convert to functional connectivity
    connectivities = signal_to_connectivities(ROI_signals, kind='correlation')
    connectivities, _ = threshold(connectivities[:2])

    # inital node embeddings
    H_0 = node_embed(connectivities)
    H_0 = normalize_features(H_0)
    # initial edge embeddings
    W_0 = torch.as_tensor(connectivities)  # TODO: implement edge_embed() function
    sparse_adj_list = sym_normalize_adj(connectivities)