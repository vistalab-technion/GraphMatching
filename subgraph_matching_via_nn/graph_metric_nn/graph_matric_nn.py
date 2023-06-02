from torch import nn

class GraphMetricNetwork(nn.Module):
    def __init__(self, loss_fun, embedding_sub):
        super().__init__()
        self._loss_fun = loss_fun
        self._embedding_sub = embedding_sub

    def forward(self, embedding_full, embedding_subgraph):
        if embedding_subgraph is not None:
            embedding_subgraph = self._embedding_sub
        else:
            raise Exception("missing sub-graph embedding")
            # TODO: compute subgraph embedding
        loss = self._loss_fun(embedding_full, embedding_subgraph)
        return loss