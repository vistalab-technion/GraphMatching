from torch import nn


class BaseGraphMetricNetwork(nn.Module):
    def __init__(self, loss_fun):
        super().__init__()
        self._loss_fun = loss_fun


class GraphMetricNetwork(BaseGraphMetricNetwork):
    def __init__(self, loss_fun):
        super().__init__(loss_fun)

    def forward(self, embedding_full, embedding_subgraph):
        loss = self._loss_fun(embedding_full, embedding_subgraph)
        return loss
