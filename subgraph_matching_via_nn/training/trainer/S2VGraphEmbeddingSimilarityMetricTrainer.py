from typing import Dict, List, Tuple

import torch
import torch_geometric as tg

from powerful_gnns.util import S2VGraph, load_data_given_graph_list_and_label_map
from subgraph_matching_via_nn.data.annotated_graph import AnnotatedGraph
from subgraph_matching_via_nn.graph_metric_networks.graph_metric_nn import BaseGraphMetricNetwork
from subgraph_matching_via_nn.training.PairSampleInfo import Pair_Sample_Info
from subgraph_matching_via_nn.training.trainer.PairSampleInfo_with_S2VGraphs import PairSampleInfo_with_S2VGraphs
from subgraph_matching_via_nn.training.trainer.SimilarityMetricTrainerBase import SimilarityMetricTrainerBase


class S2VGraphEmbeddingSimilarityMetricTrainer(SimilarityMetricTrainerBase):
    def __init__(self, graph_similarity_module: BaseGraphMetricNetwork, dump_base_path: str,
                 problem_params: Dict, solver_params: Dict):
        super(S2VGraphEmbeddingSimilarityMetricTrainer, self).__init__(graph_similarity_module, dump_base_path,
                                                      problem_params, solver_params)

    @staticmethod
    def get_model_expected_input_dim(sample_pair: Pair_Sample_Info):
        return sample_pair.masked_graph.node_indicator.shape[1]

    def __convert_annotated_graph_into_s2vgraph(self, annotated_graph: AnnotatedGraph):
        # nx.Graph -> S2VGraph
        s2v_graph = S2VGraph(annotated_graph.g, label=None)
        batch_graph, _ = load_data_given_graph_list_and_label_map([s2v_graph], label_dict = {}, degree_as_tag=True,
                                                                  device=self.device, print_stats=False)

        # node_features: w
        # need to override the node_features
        batch_graph[0].node_features = annotated_graph.node_indicator.to(device=self.device)

        return batch_graph[0]

    def __convert_pair_sample_info_list_to_s2vgraph_tuple_list(self, pair_sample_info_list: List[Pair_Sample_Info]):
        return [
            (
                self.__convert_annotated_graph_into_s2vgraph(pair_sample_info.masked_graph),
                self.__convert_annotated_graph_into_s2vgraph(pair_sample_info.subgraph)
             ) for pair_sample_info in pair_sample_info_list]

    def get_data_loaders(self, train_set: List[Pair_Sample_Info], val_set: List[Pair_Sample_Info],
                         new_samples_amount) -> (tg.data.DataLoader, tg.data.DataLoader):
        train_s2vgraph_tuple_list = self.__convert_pair_sample_info_list_to_s2vgraph_tuple_list(train_set)
        val_s2vgraph_tuple_list = self.__convert_pair_sample_info_list_to_s2vgraph_tuple_list(val_set)

        combined_types_train_set = [PairSampleInfo_with_S2VGraphs(pair_sample_info, s2v_graphs) for pair_sample_info, s2v_graphs in zip(train_set, train_s2vgraph_tuple_list)]
        combined_types_val_set = [PairSampleInfo_with_S2VGraphs(pair_sample_info, s2v_graphs) for pair_sample_info, s2v_graphs in zip(val_set, val_s2vgraph_tuple_list)]

        train_loader, val_loader = self._build_data_loaders(combined_types_train_set, combined_types_val_set,
                                                            S2VGraphEmbeddingSimilarityMetricTrainer.custom_collate)

        return train_loader, val_loader

    @staticmethod
    def custom_collate(samples_batch):
        pair_sample_info_samples_batch = [sample.pair_sample_info for sample in samples_batch]
        collated_samples_batch, _ = SimilarityMetricTrainer.custom_collate(pair_sample_info_samples_batch)

        return collated_samples_batch, samples_batch

    def _get_pairs_list_loss(self, batch: Tuple[Pair_Sample_Info, List[PairSampleInfo_with_S2VGraphs]]) -> torch.Tensor:
        collated_samples, samples = batch

        embeddings_metric_loss = self.graph_similarity_module.forward(
            [sample.s2v_graphs for sample in samples]
        )

        return self.get_aggregated_pairs_batch_distance(embeddings_metric_loss, (collated_samples, [elem.pair_sample_info for elem in samples]))