import networkx as nx

from subgraph_matching_via_nn.data.annotated_graph import AnnotatedGraph
from subgraph_matching_via_nn.training.PairSampleBase import PairSampleBase


class PairSampleBase:
    localization_object: object  # None if pair was generated without any optimization involved

    def __init__(self,
                 masked_graph: AnnotatedGraph,
                 subgraph: AnnotatedGraph,
                 is_negative_sample: bool = None,
                 localization_object: bool = None
                 ):
        super().__init__()
        self.masked_graph = masked_graph
        self.subgraph = subgraph
        self.is_negative_sample = is_negative_sample
        self.localization_object = localization_object


class Pair_Sample_Info(PairSampleBase):
    # graph is the source graph; subgraph is the target graph (the searched graph)

    def __init__(self,
                 masked_graph: AnnotatedGraph,
                 subgraph: AnnotatedGraph,
                 is_negative_sample = None):
        super().__init__(masked_graph=masked_graph,
                         subgraph=subgraph,
                         is_negative_sample=is_negative_sample)
