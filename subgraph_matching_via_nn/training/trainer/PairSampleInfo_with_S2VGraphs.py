from typing import Tuple

from powerful_gnns.util import S2VGraph
from subgraph_matching_via_nn.training.PairSampleInfo import Pair_Sample_Info


class PairSampleInfo_with_S2VGraphs:
    def __init__(self, pair_sample_info: Pair_Sample_Info, s2v_graphs: Tuple[S2VGraph, S2VGraph]):
        self.pair_sample_info = pair_sample_info
        self.s2v_graphs = s2v_graphs