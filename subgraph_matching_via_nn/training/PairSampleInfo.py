from typing import Optional

from subgraph_matching_via_nn.composite_nn.localization_state_replayer import ReplayableLocalizationState, \
    LocalizationStateReplayer
from subgraph_matching_via_nn.data.annotated_graph import AnnotatedGraph


class PairSampleBase:
    localization_object: Optional[LocalizationStateReplayer]
    # None if pair was generated without any optimization involved, or was not filled yet by trainer module

    localization_state_object: Optional[ReplayableLocalizationState]
    # None if pair was generated without any optimization involved

    def __init__(self,
                 masked_graph: AnnotatedGraph,
                 subgraph: AnnotatedGraph,
                 is_negative_sample: bool = None,
                 localization_state_object: ReplayableLocalizationState = None
                 ):
        super().__init__()
        self.masked_graph = masked_graph
        self.subgraph = subgraph
        self.is_negative_sample = is_negative_sample
        self.localization_state_object = localization_state_object
        self.localization_object = None


class Pair_Sample_Info(PairSampleBase):
    # graph is the source graph; subgraph is the target graph (the searched graph)

    def __init__(self,
                 masked_graph: AnnotatedGraph,
                 subgraph: AnnotatedGraph,
                 is_negative_sample = None):
        super().__init__(masked_graph=masked_graph,
                         subgraph=subgraph,
                         is_negative_sample=is_negative_sample)
