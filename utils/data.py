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
    Load the Saved 3D ROI signals
    Params :
    --------
        - correlation_matrices (m * n * n np.array) : the weighted adjacency matrices
        - threshold (float) : percentage
    Returns :
        - adjacency matrix in n * n format, with diagonal 0
        - list of nx.graph
    '''
    # Creates graph using the data of the correlation matrix
    graph_list = []
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


def node_embed(correlation_matrices, mask_name='AAL', hand_crafted=True):
    '''
    embed each node
    Params :
    --------
        - correlation_matrices (m * n * n np.array) : the weighted adjacency matrices
        - hand_crafted (bool) : using hand_engineered feature or vector embedding
    Returns :
        - adjacency matrix in (n_subjects, ) format, with diagonal 0
        TODO: Documentation + allow node2Vec embed
    '''
    coordinate = torch.tensor(np.load('../data/' + mask_name + "_coordinates.npy", allow_pickle=True), dtype=torch.float)
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
    return torch.stack([x.float() for x in H], dim=0)


def signal_to_connectivities(signals, kind='correlation', discard_diagonal=True, vectorize=False):
    '''
    extract connectivities from time series signals
    Params :
    --------
        - signals ((num_subject, time_frame, num_ROI) np.array) : the ROI signals
        - TODO: documentation
    Returns :
    --------
        - the functional connectivity
        - type : numpy matrix or numpy vector
    '''
    # define a correlation measure
    correlation_measure = ConnectivityMeasure(kind=kind, discard_diagonal=discard_diagonal, vectorize=vectorize)
    # transform to connectivity matrices
    functional_connectivity = correlation_measure.fit_transform(signals)

    return functional_connectivity


def load_fmri_data(dataDir='../data', dataset='271_AAL', connectivity=True, verbose=True):
    '''
    Load the Saved 3D ROI signals
    Params :
    --------
        - dataDir (str) : the path of the data directory
        - dataset (str) : the name of the dataset
        - connectivity (bool) :
    Returns :
        - the signals of brain
        TODO: documentation
    '''
    subjects_list = np.load(dataDir + "/" + dataset + ".npy", allow_pickle=True)
    label_list = np.load(dataDir + "/" + dataset + "_label.npy", allow_pickle=True)
    classes, classes_idx, classes_count = np.unique(label_list, return_inverse=True, return_counts=True)

    if connectivity:
        subjects_list = signal_to_connectivities(subjects_list)

    if verbose:
        # TODO: print the information
        print(classes)
        print(classes_count)

    return subjects_list, label_list, classes_idx


def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


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


def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1))  # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten()  # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0.  # zero inf data
    r_mat_inv = sp.diags(r_inv)  # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features)  # D^-1:[2708, 2708]@X:[2708, 2708]
    return sparse_to_tuple(features)  # [coordinates, data, shape], []


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def chebyshev_polynomials(adj, k):
    """
    Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation).
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
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
    connectivities, labels, labels_idex = load_fmri_data()
    connectivities, _ = threshold(connectivities[:3])
    # inital node embeddings
    H_0 = node_embed(connectivities)


