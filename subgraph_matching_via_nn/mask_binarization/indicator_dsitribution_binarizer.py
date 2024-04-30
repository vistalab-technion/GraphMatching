from enum import Enum
import kmeans1d
import networkx as nx
import numpy as np
import scipy as sp

from subgraph_matching_via_nn.mask_binarization.LP_binarization import solve_maximum_weight_subgraph
from subgraph_matching_via_nn.utils.graph_utils import graph_edit_matrix
from subgraph_matching_via_nn.utils.utils import NP_DTYPE, top_m


class IndicatorBinarizationBootType(Enum):
    OptimalElement = 0,
    SeriesNormalizedMean = 1,
    SeriesMedianOfBinarizedElements = 2,


class IndicatorBinarizationType(Enum):
    KMeans = 0,
    TopK = 1,
    Quantile = 2,
    Diffusion = 3,
    zoomout = 4,
    nonlinear_zoomout = 5,
    mwksp = 6,


class IndicatorDistributionBinarizer:

    @staticmethod
    def binarize(graph: nx.graph, w: np.array, params, type: IndicatorBinarizationType, as_dict:bool=True):
        if type == IndicatorBinarizationType.KMeans:
            w_th, centroids = kmeans1d.cluster(w, k=2)
            w_th = np.array(w_th)[:, None]
        elif type == IndicatorBinarizationType.TopK:
            w_th = top_m(w, params["m"])
        elif type == IndicatorBinarizationType.Quantile:
            w_th = (w > np.quantile(w, params["quantile_level"]))
            w_th = np.array(w_th, dtype=np.float64)
        elif type == IndicatorBinarizationType.Diffusion:
            A = (nx.adjacency_matrix(graph)).toarray()
            D = np.diag(A.sum(axis=1))
            L = D - A
            # # Eigenvalue decomposition of the Laplacian
            # eigenvalues, eigenvectors = np.linalg.eigh(L)

            # Generalized eigenvalue decomposition of the Random Walk Laplacian
            # eigenvalues, eigenvectors = np.linalg.eigh(L)
            eigenvalues, eigenvectors = sp.linalg.eigh(L, D)

            k = 10

            # Generate k logarithmically spaced values of t from a large to small value
            max_t = 10  # Change this to your desired maximum value
            min_t = 0.01  # Change this to your desired minimum value
            t_values = np.logspace(np.log10(max_t), np.log10(min_t), k)

            w_th = w
            for t in t_values:
                # Apply the heat kernel using matrix exponentiation
                heat_matrix = eigenvectors @ np.diag(
                    np.exp(-t * eigenvalues)) @ eigenvectors.T
                heat_w = heat_matrix @ w_th

                # Binarize by keeping the largest m components
                w_th = top_m(heat_w, params["m"])
        elif type == IndicatorBinarizationType.zoomout:
            A = (nx.adjacency_matrix(graph)).toarray()
            D = np.diag(A.sum(axis=1))
            L = D - A

            # Generalized eigenvalue decomposition of the Random Walk Laplacian
            # eigenvalues, eigenvectors = np.linalg.eigh(L)
            eigenvalues, eigenvectors = sp.linalg.eigh(L, D)

            w_th = w
            for i in range(2, A.shape[0]):
                # Apply the heat kernel using matrix exponentiation
                heat_w = eigenvectors[:, :i] @ eigenvectors[:, :i].T @ w_th

                # Binarize by keeping the largest m components
                w_th = top_m(heat_w, params["m"])

        elif type == IndicatorBinarizationType.nonlinear_zoomout:
            A = (nx.adjacency_matrix(graph)).toarray()
            D = np.diag(A.sum(axis=1))
            L = D - A
            w_th = w
            heat_w = w
            for i in range(2, A.shape[0]):
                E = graph_edit_matrix(A, 1 - params["m"] * w_th)
                Ae = A - E

                De = np.diag(Ae.sum(axis=1))
                Le = De - Ae

                # Generalized eigenvalue decomposition of the Random Walk Laplacian
                eigenvalues, eigenvectors = np.linalg.eigh(Le)
                # eigenvalues, eigenvectors = sp.linalg.eigh(Le, De)
                # Apply the heat kernel using matrix exponentiation
                heat_w = eigenvectors[:, :i] @ eigenvectors[:, :i].T @ w_th

                # Binarize by keeping the largest m components
                w_th = top_m(heat_w, params["m"])
        elif type == IndicatorBinarizationType.mwksp:
            num_nodes = params['m']
            num_edges = params['n']

            A = (nx.adjacency_matrix(graph)).toarray()
            selected_nodes, selected_edges = solve_maximum_weight_subgraph(w, A, num_nodes, num_edges)
            print(f'requested: n_nodes = {num_nodes}, n_edges : {num_edges}')
            print(f'found: n_nodes = {len(selected_nodes)}, n_edges : {len(selected_edges)}')
            w_th = np.zeros([len(graph.nodes()), 1])
            w_th[selected_nodes] = 1.0
        else:
            w_th = w

        w_th = w_th / w_th.sum()

        if as_dict:
            return dict(zip(graph.nodes(), w_th))
        return w_th

    @staticmethod
    def from_indicators_series_to_binary_indicator(processed_G, w_all, w_star, params,
                                                   series_binarization_type: IndicatorBinarizationBootType,
                                                   element_binarization_type: IndicatorBinarizationType):

        binarize = IndicatorDistributionBinarizer.binarize

        if series_binarization_type == IndicatorBinarizationBootType.OptimalElement:
            return binarize(processed_G, w_star, params, element_binarization_type)
        elif series_binarization_type == IndicatorBinarizationBootType.SeriesNormalizedMean:
            w_boot = np.mean(np.array(w_all), axis=0)
            w_boot = binarize(processed_G, w_boot, params, None, as_dict=False)
            return binarize(processed_G, w_boot, params, element_binarization_type)
        elif series_binarization_type == IndicatorBinarizationBootType.SeriesMedianOfBinarizedElements:
            return binarize(processed_G, np.median(
                np.array([list(binarize(processed_G, w, params, element_binarization_type).values()) for w in w_all]), axis=0),
                            params, element_binarization_type)
        else:
            raise ValueError(f"Unsupported series binarization type: {series_binarization_type}")
