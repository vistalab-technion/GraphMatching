# -----------------------------------------------------------------------------
# This file contains the API for the WWL kernel computations
#
# December 2019, M. Togninalli
# -----------------------------------------------------------------------------
import contextlib
from os import cpu_count

from joblib import delayed, Parallel
from sklearn.metrics.pairwise import check_pairwise_arrays, manhattan_distances
from tqdm import tqdm

from common.parallel_computation import tqdm_joblib
from .propagation_scheme import WeisfeilerLehman, ContinuousWeisfeilerLehman
import ot
import numpy as np
import itertools as it


def _compute_wasserstein_distance(label_sequences, label_index_to_sequence_for_comparison_map, sinkhorn=False,
                                    categorical=False, sinkhorn_lambda=1e-2):
    '''
    Generate the Wasserstein distance matrix for the graphs embedded 
    in label_sequences
    '''
    # Get the iteration number from the embedding file
    n1 = len(label_sequences)
    n2 = len(label_index_to_sequence_for_comparison_map)

    M = np.zeros((n1,n2))
    min_indices_graph_2 = min(label_index_to_sequence_for_comparison_map.items())
    # relative_indices_graph_2 = []
    sorted_indices_graph_2 = sorted(label_index_to_sequence_for_comparison_map.keys(), reverse=False)
    # for graph_index_2 in sorted_indices_graph_2:
    #     relative_indices_graph_2
    # - min_indices_graph_2

    # Iterate over pairs of graphs
    for graph_index_1, graph_1 in enumerate(label_sequences):
        # Only keep the embeddings for the first h iterations
        labels_1 = label_sequences[graph_index_1]
        for graph_index_2, _ in label_index_to_sequence_for_comparison_map.items():
            if graph_index_2 < graph_index_1:
                continue
            labels_2 = label_sequences[graph_index_2]
            M_index_2 = sorted_indices_graph_2.index(graph_index_2)

            # Get cost matrix
            ground_distance = 'hamming' if categorical else 'euclidean'
            costs = ot.dist(labels_1, labels_2, metric=ground_distance)

            if sinkhorn:
                mat = ot.sinkhorn(np.ones(len(labels_1))/len(labels_1), 
                                    np.ones(len(labels_2))/len(labels_2), costs, sinkhorn_lambda, 
                                    numItermax=50)
                M[graph_index_1, M_index_2] = np.sum(np.multiply(mat, costs))
            else:
                M[graph_index_1, M_index_2] = \
                    ot.emd2([], [], costs)
                    
    # M = (M + M.T)
    return M

def pairwise_wasserstein_distance(X, x_indices_for_comparison, node_features = None, num_iterations=3, sinkhorn=False, enforce_continuous=False):
    """
    Pairwise computation of the Wasserstein distance between embeddings of the 
    graphs in X.
    args:
        X (List[ig.graphs]): List of graphs
        node_features (array): Array containing the node features for continuously attributed graphs
        num_iterations (int): Number of iterations for the propagation scheme
        sinkhorn (bool): Indicates whether sinkhorn approximation should be used
    """
    # First check if the graphs are continuous vs categorical
    categorical = True
    if enforce_continuous:
        print('Enforce continous flag is on, using CONTINUOUS propagation scheme.')
        categorical = False
    elif node_features is not None:
        print('Continuous node features provided, using CONTINUOUS propagation scheme.')
        categorical = False
    else:
        for g in X:
            if not 'label' in g.vs.attribute_names():
                print('No label attributed to graphs, use degree instead and use CONTINUOUS propagation scheme.')
                categorical = False
                break
        if categorical:
            print('Categorically-labelled graphs, using CATEGORICAL propagation scheme.')
    
    # Embed the nodes
    if categorical:
        es = WeisfeilerLehman()
        node_representations = es.fit_transform(X, num_iterations=num_iterations)
    else:
        es = ContinuousWeisfeilerLehman()
        node_representations = es.fit_transform(X, node_features=node_features, num_iterations=num_iterations)

    # Compute the Wasserstein distance
    node_for_comparison_representations = {index: X[index] for index in x_indices_for_comparison}
    pairwise_distances = _compute_wasserstein_distance(node_representations, node_for_comparison_representations, sinkhorn=sinkhorn,
                                    categorical=categorical, sinkhorn_lambda=1e-2)
    return pairwise_distances

def range_manhattan_distances(X, Y, range_X, range_Y):
    ranged_X = X[range_X[0]: range_X[1] + 1]
    ranged_Y = Y[range_Y[0]: range_Y[1] + 1]

    dist = manhattan_distances(ranged_X, ranged_Y)

    print(f"finished distance calculation for range X {range_X} and range Y {range_Y}", flush=True)

    return range_X, range_Y, dist

def laplacian_kernel(X, Y=None, gamma=None):
    """Compute the laplacian kernel between X and Y.

    The laplacian kernel is defined as::

        K(x, y) = exp(-gamma ||x-y||_1)

    for each pair of rows x in X and y in Y.
    Read more in the :ref:`User Guide <laplacian_kernel>`.

    .. versionadded:: 0.17

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)

    Y : ndarray of shape (n_samples_Y, n_features), default=None

    gamma : float, default=None
        If None, defaults to 1.0 / n_features.

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    range_chunk_size = 1000
    pairs_indices_x_ranges = np.array_split(range(len(X)), len(X) // range_chunk_size)
    pairs_indices_x_ranges = np.array([np.array(
        [pairs_indices_x_range[0], pairs_indices_x_range[-1]]) for pairs_indices_x_range in pairs_indices_x_ranges])
    pairs_indices_y_ranges = np.array_split(range(len(Y)), len(Y) // range_chunk_size)
    pairs_indices_y_ranges = np.array([np.array([pairs_indices_y_range[0], pairs_indices_y_range[-1]]) for pairs_indices_y_range in pairs_indices_y_ranges])

    pairs_indices_ranges = np.array(list(it.product(pairs_indices_x_ranges, pairs_indices_y_ranges)))

    with tqdm_joblib(tqdm(desc="My calculation", total=len(pairs_indices_ranges))) as progress_bar:
        ranged_distances = Parallel(n_jobs=int(cpu_count()), prefer='processes')(
            delayed(range_manhattan_distances)(X=X, Y=Y, range_X=pairs_indices_range[0], range_Y=pairs_indices_range[1])
            for pairs_indices_range in pairs_indices_ranges
        )

    distances = np.empty(shape=(len(X), len(Y)))
    for range_X, range_Y, dist in ranged_distances:
        distances[range_X[0]: range_X[1] + 1, range_Y[0]: range_Y[1] + 1] = dist

    K = -gamma * distances
    np.exp(K, K)  # exponentiate K in-place
    return K

def wwl(X, node_features=None, num_iterations=3, sinkhorn=False, gamma=None):
    """
    Pairwise computation of the Wasserstein Weisfeiler-Lehman kernel for graphs in X.
    """
    D_W =  pairwise_wasserstein_distance(X, node_features = node_features, 
                                num_iterations=num_iterations, sinkhorn=sinkhorn)
    wwl = laplacian_kernel(D_W, gamma=gamma)
    return wwl


#######################
# Class implementation
#######################
