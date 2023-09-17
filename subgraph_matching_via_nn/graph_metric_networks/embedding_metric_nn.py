from typing import List

import torch
from torch import nn
from torch.nn.modules.loss import _Loss


class BaseEmbeddingMetricNetwork(nn.Module):
    def __init__(self, loss_fun: _Loss):
        super().__init__()
        self._loss_fun = loss_fun


class EmbeddingMetricNetwork(BaseEmbeddingMetricNetwork):
    def __init__(self, loss_fun: _Loss):
        """

        :param loss_fun: a torch loss function
        """
        super().__init__(loss_fun)

    def forward(self, embeddings_full, embeddings_subgraph):
        """
        :param embeddings_full: either a list of embedding tensors or a single tensor
        of the full graph
        :param embeddings_subgraph: either a list of embedding tensors or a single tensor
        of the sub graph
        :return: loss between embeddings
        """
        if isinstance(embeddings_full, list):
            loss = 0.0
            for embedding_full, embedding_subgraph in zip(embeddings_full,
                                                          embeddings_subgraph):
                loss += self._loss_fun(embedding_full, embedding_subgraph)
        elif isinstance(embeddings_full, torch.Tensor):
            loss = self._loss_fun(embeddings_full, embeddings_subgraph)
        else:
            raise (f'embeddings of type {embeddings_full.type} not supported.')
        return loss
