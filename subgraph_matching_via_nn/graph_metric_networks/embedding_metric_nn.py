from typing import List

import torch
from torch import nn
from torch.nn.modules.loss import _Loss


class BaseEmbeddingMetricNetwork(nn.Module):
    def __init__(self, loss_fun: _Loss):
        super().__init__()
        self._loss_fun = loss_fun


class EmbeddingMetricNetwork(BaseEmbeddingMetricNetwork):
    def __init__(self, loss_fun: _Loss, params: dict):
        """

        :param loss_fun: a torch loss function
        """
        super().__init__(loss_fun)
        self.params = params

    def forward(self, embeddings_full, embeddings_subgraph, is_sum_tensor_into_single_item=True):
        """
        :param embeddings_full: either a list of embedding tensors or a single tensor
        of the full graph
        :param embeddings_subgraph: either a list of embedding tensors or a single tensor
        of the sub graph
        :param is_sum_tensor_into_single_item: False for returning a tensor of pairs losses;
        True for returning the sum of pair losses as a single element
        :return: loss between embeddings
        """

        scaler = self.params['scaler']

        if isinstance(embeddings_full, list):
            losses = [self._loss_fun(embedding_full / scaler, embedding_subgraph / scaler)
                      for embedding_full, embedding_subgraph in zip(embeddings_full, embeddings_subgraph)]
            losses = torch.stack(losses)

            if is_sum_tensor_into_single_item:
                loss = losses.sum()
            else:
                return losses
        elif isinstance(embeddings_full, torch.Tensor):
            loss = self._loss_fun(embeddings_full, embeddings_subgraph)
        else:
            raise (f'embeddings of type {embeddings_full.type} not supported.')
        return loss
