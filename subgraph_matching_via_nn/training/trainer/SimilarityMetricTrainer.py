from typing import Dict, List, Tuple

import torch
import torch_geometric as tg
from torch_geometric.loader.dataloader import Collater

from subgraph_matching_via_nn.graph_metric_networks.graph_metric_nn import BaseGraphMetricNetwork
from subgraph_matching_via_nn.training.PairSampleInfo import Pair_Sample_Info
from subgraph_matching_via_nn.training.trainer.SimilarityMetricTrainerBase import SimilarityMetricTrainerBase


class SimilarityMetricTrainer(SimilarityMetricTrainerBase):
    def __init__(self, graph_similarity_module: BaseGraphMetricNetwork, dump_base_path: str,
                 problem_params: Dict, solver_params: Dict):
        super(SimilarityMetricTrainer, self).__init__(graph_similarity_module, dump_base_path,
                                                      problem_params, solver_params)


    def get_data_loaders(self, train_set: List[Pair_Sample_Info],
                         val_set: List[Pair_Sample_Info], new_samples_amount, device_ids) \
            -> (tg.data.DataLoader, tg.data.DataLoader):

        train_loader, val_loader = self._build_data_loaders(train_set, val_set,
                                                            SimilarityMetricTrainer.custom_collate, device_ids, new_samples_amount)

        return train_loader, val_loader

    @staticmethod
    def custom_collate(samples_batch):
        collator = Collater([], exclude_keys=None)

        is_negative_sample = collator.collate(
            [sample.is_negative_sample for sample in samples_batch])

        # use graph types (instead of S2VGrapn and nx.graph) that support batching
        # collated_graph = collator.collate([sample.graph.to(device=device) for sample in samples_batch])
        # collated_masked_graph = collator.collate([sample.masked_graph.to(device=device) for sample in samples_batch])
        # collated_subgraph = collator.collate([sample.subgraph.to(device=device) for sample in samples_batch])
        collated_graph = None
        collated_masked_graph = None
        collated_subgraph = None

        collated_samples_batch = Pair_Sample_Info(masked_graph=collated_masked_graph,
                                                  subgraph=collated_subgraph,
                                                  is_negative_sample=is_negative_sample)

        return collated_samples_batch, samples_batch


    def _get_pairs_list_loss(self, batch: Tuple[Pair_Sample_Info, List[Pair_Sample_Info]]) -> torch.Tensor:
        _, samples = batch

        embeddings_metric_loss = self.graph_similarity_module.forward(
            [(sample.masked_graph, sample.subgraph) for sample in samples]
        )

        return self.get_aggregated_pairs_batch_distance(embeddings_metric_loss, batch)