from enum import Enum
import kmeans1d
import networkx as nx
import numpy as np
from subgraph_matching_via_nn.utils.utils import NP_DTYPE


class IndicatorBinarizationBootType(Enum):
    OptimalElement = 0,
    SeriesNormalizedMean = 1,
    SeriesMedianOfBinarizedElements = 2,


class IndicatorBinarizationType(Enum):
    KMeans = 0,
    TopK = 1,
    Quantile = 2,


class IndicatorDistributionBinarizer:

    @staticmethod
    def binarize(graph: nx.graph, w: np.array, params, type: IndicatorBinarizationType, as_dict=True):
        if type == IndicatorBinarizationType.KMeans:
            w_th, centroids = kmeans1d.cluster(w, k=2)
            w_th = np.array(w_th)[:, None]
        elif type == IndicatorBinarizationType.TopK:
            indices_of_top_m = np.argsort(w, axis=0)[-params["m"]:]  # top m
            w_th = np.zeros_like(w, dtype=NP_DTYPE)
            w_th[indices_of_top_m] = 1
        elif type == IndicatorBinarizationType.Quantile:
            w_th = (w > np.quantile(w, params["quantile_level"]))
            w_th = np.array(w_th, dtype=np.float64)
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
