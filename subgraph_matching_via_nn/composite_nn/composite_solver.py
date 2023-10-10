from typing import Optional, Tuple, List
import networkx as nx
import numpy as np
import torch
from torch import optim, nn
from livelossplot import PlotLosses
from subgraph_matching_via_nn.composite_nn.composite_nn import CompositeNeuralNetwork
from subgraph_matching_via_nn.data.sub_graph import SubGraph
from subgraph_matching_via_nn.graph_metric_networks.embedding_metric_nn import EmbeddingMetricNetwork
from subgraph_matching_via_nn.graph_processors.graph_processors import BaseGraphProcessor, GraphProcessor
from subgraph_matching_via_nn.mask_binarization.indicator_dsitribution_binarizer import IndicatorDistributionBinarizer
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE, uniform_dist


class BaseCompositeSolver(nn.Module):

    def __init__(self, composite_nn: CompositeNeuralNetwork, embedding_metric_nn: EmbeddingMetricNetwork,
                 graph_processor: Optional[BaseGraphProcessor] = GraphProcessor(), params: dict = {}):
        super().__init__()
        self.composite_nn = composite_nn
        self.embedding_metric_nn = embedding_metric_nn
        self.graph_processor = graph_processor
        self.params = params

        self.liveloss = PlotLosses(mode='notebook')

    def compare_indicators(self, A_full_processed, indicator_name_to_object_map: dict, embedding_id):
        for indicator_name, indicator_object in indicator_name_to_object_map.items():
            embedding_nn = self.composite_nn.embedding_networks[embedding_id]
            indicator_embedding = embedding_nn(w=torch.tensor(indicator_object, requires_grad=False),
                                               A=A_full_processed.detach()).type(TORCH_DTYPE)
            print(
                f"{[f'{value:.4f}' for value in indicator_embedding]} : {indicator_name} {embedding_nn.embedding_type}")

    def compare(self, A_full_processed, A_sub_processed, gt_indicator_tensor, A_sub_indicator=None):
        if A_sub_indicator is None:
            A_sub_indicator = uniform_dist(A_sub_processed.shape[0]).detach()
        embeddings_sub = self.composite_nn.embed(A=A_sub_processed.detach().type(TORCH_DTYPE),
                                            w=A_sub_indicator)

        embeddings_full, w = self.composite_nn(A=A_full_processed, params=self.params)
        loss = self.embedding_metric_nn(embeddings_full=embeddings_full,
                                   embeddings_subgraph=embeddings_sub)

        embeddings_gt = self.composite_nn.embed(A=A_full_processed.detach().type(TORCH_DTYPE),
                                           w=gt_indicator_tensor)
        ref_loss = self.embedding_metric_nn(embeddings_gt, embeddings_sub)


        embedding_nns = self.composite_nn.embedding_networks
        embedding_nns_amount = len(embedding_nns)
        if len(embeddings_full) != embedding_nns_amount:
            print(
                f"{[f'{value:.4f}' for value in embeddings_full[0]]} : init")
            print(
                f"{[f'{value:.4f}' for value in embeddings_sub[0]]} : sub")
            print(
                f"{[f'{value:.4f}' for value in embeddings_gt[0]]} : GT")
        else:
            for idx in range(embedding_nns_amount):
                print(
                    f"{[f'{value:.4f}' for value in embeddings_full[idx]]} : init {embedding_nns[idx].embedding_type}")
                print(
                    f"{[f'{value:.4f}' for value in embeddings_sub[idx]]} : sub {embedding_nns[idx].embedding_type}")
                print(
                    f"{[f'{value:.4f}' for value in embeddings_gt[idx]]} : GT {embedding_nns[idx].embedding_type}")

        print(f"init loss (no reg): {loss}")  # without regularization

        reg = self._get_reg_loss(A_full_processed, w)
        full_loss = loss + reg
        print(f"init full loss (with reg): {full_loss}")  # with regularization

        return loss, ref_loss

    def _get_reg_loss(self, A_full_processed, w):
        reg_terms_list = [reg_param * reg_term(A_full_processed, w, self.params) for reg_param, reg_term in
                           zip(self.params["reg_params"], self.params["reg_terms"])]

        if len(reg_terms_list) == 0:
            return 0
        return torch.stack(reg_terms_list).sum()

    def get_composite_loss_terms(self, A, embeddings_sub):
        x0 = self.params.get("x0", None)
        embeddings_full, w = self.composite_nn(A, x0, self.params)

        loss = self.embedding_metric_nn(embeddings_full=embeddings_full,
                                        embeddings_subgraph=embeddings_sub)

        reg = self._get_reg_loss(A, w)
        return loss, reg

    def __log_loss(self, iteration, loss, reg):
        if iteration % self.params['k_update_plot'] == 0:
            full_loss = loss + reg

            print(f"Iteration {iteration}, Loss: {loss.item()}")
            print(f"Iteration {iteration}, Reg: {reg.item()}")
            print(f"Iteration {iteration}, Loss + rho * Reg: {full_loss.item()}")
            self.liveloss.update({'loss': loss.item()})
            self.liveloss.send()

    def _create_optimizer(self):
        lr = self.params['lr']
        solver_type = self.params.get("solver_type", None)
        if solver_type == 'gd':
            optimizer = optim.SGD(params=self.composite_nn.parameters(), lr=lr)
        elif solver_type == 'lbfgs':
            optimizer = optim.LBFGS(params=self.composite_nn.parameters(), lr=lr, max_iter=5,
                                    max_eval=None,
                                    tolerance_grad=1e-07,
                                    tolerance_change=1e-09,
                                    history_size=10,
                                    line_search_fn=None)
        else:
            raise ValueError(f"Unknown optimizer choice: {solver_type}")
        return optimizer

    def __pre_process_graphs(self, G: nx.graph, G_sub: nx.graph):
        # preprocess the graphs, e.g. to get a line-graph
        G = self.graph_processor.pre_process(SubGraph(G))
        G_sub = self.graph_processor.pre_process(SubGraph(G_sub))
        sub_graph = SubGraph(G, G_sub)
        A = sub_graph.A_full
        A_sub = sub_graph.A_sub

        return A, A_sub

    def forward(self, input_graphs):
        """
        input_graphs (List[Tuple[nx.graph, nx.graph]])
        """
        #TODO: optimize with batching
        graphs_distances_list = []
        for G, G_sub in input_graphs:
            graphs_distance = self.get_loss_for_graph_and_subgraph(G, G_sub)
            graphs_distances_list.append(graphs_distance)
        return torch.stack(graphs_distances_list).unsqueeze(1)

    def get_loss_for_graph_and_subgraph(self, G: nx.graph, G_sub: nx.graph, dtype=torch.double):
        A, A_sub = self.__pre_process_graphs(G, G_sub)
        embeddings_sub = self.composite_nn.embed(A=A_sub.detach().type(dtype),
                                                 w=uniform_dist(A_sub.shape[0]).detach())
        loss, reg = self.get_composite_loss_terms(A, embeddings_sub)
        return loss + reg

    def solve(self, G: nx.graph, G_sub: nx.graph, dtype=torch.double):
        A, A_sub = self.__pre_process_graphs(G, G_sub)
        embeddings_sub = self.composite_nn.embed(A=A_sub.detach().type(dtype),
                                                 w=uniform_dist(A_sub.shape[0]).detach())

        self.composite_nn.train() # Set the model to training mode
        optimizer = self._create_optimizer()

        for iteration in range(self.params["maxiter"]):  # TODO: add stopping condition
            loss, reg = self.get_composite_loss_terms(A, embeddings_sub)
            full_loss = loss + reg

            optimizer.zero_grad()
            full_loss.backward()
            optimizer.step()

            self.__log_loss(iteration, loss, reg)

        w_star = self.composite_nn.node_classifier_network(A=A, params=self.params).detach().numpy()
        return w_star

    def set_initial_params_based_on_previous_optimum(self, w_star):
        # binarized_w_star = IndicatorDistributionBinarizer.binarize(processed_G, w_star, self.params,
        #                                  binarization_type)
        # w_th = torch.tensor(list(binarized_w_star.values()))

        x0 = w_star-np.min(w_star)
        x0 = torch.tensor(x0 / x0.sum())
        self.composite_nn.node_classifier_network.init_params(default_weights=x0)
        return x0