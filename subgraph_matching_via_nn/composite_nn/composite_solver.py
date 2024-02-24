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
        device = self.params['device']

        A_full_processed = A_full_processed.to(device=device)
        for indicator_name, indicator_object in indicator_name_to_object_map.items():
            embedding_nn = self.composite_nn.embedding_networks[embedding_id]

            indicator_embedding = embedding_nn(w=torch.tensor(indicator_object, requires_grad=False, device=device),
                                               A=A_full_processed.detach()).type(TORCH_DTYPE)
            torch.set_printoptions(precision=4)
            print(
                f"{[value for value in indicator_embedding]} : {indicator_name} {embedding_nn.embedding_type}")

    def compare(self, A_full_processed, A_sub_processed, gt_indicator_tensor, A_sub_indicator=None, print_embeddings=True):
        device = self.params['device']
        if A_sub_indicator is None:
            A_sub_indicator = uniform_dist(A_sub_processed.shape[0]).detach()
        A_sub_indicator = A_sub_indicator.to(device=device)
        gt_indicator_tensor = gt_indicator_tensor.to(device=device)
        A_sub_processed = A_sub_processed.to(device=device)
        A_full_processed = A_full_processed.to(device=device)

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

        if print_embeddings:

            torch.set_printoptions(precision=4)

            if len(embeddings_full) != embedding_nns_amount:
                print(
                    f"{[value for value in embeddings_full[0]]} : init")
                print(
                    f"{[value for value in embeddings_sub[0]]} : sub")
                print(
                    f"{[value for value in embeddings_gt[0]]} : GT")
            else:
                for idx in range(embedding_nns_amount):
                    print(
                        f"{[value for value in embeddings_full[idx]]} : init {embedding_nns[idx].embedding_type}")
                    print(
                        f"{[value for value in embeddings_sub[idx]]} : sub {embedding_nns[idx].embedding_type}")
                    print(
                        f"{[value for value in embeddings_gt[idx]]} : GT {embedding_nns[idx].embedding_type}")

        print(f"init loss (no reg): {loss}")  # without regularization
        reg = self._get_reg_loss(A_full_processed, w)
        full_loss = loss + reg
        print(f"init full loss (with reg): {full_loss}")  # with regularization

        print(f"ref loss (no reg): {ref_loss}")  # without regularization
        ref_loss_reg = self._get_reg_loss(A_full_processed, gt_indicator_tensor)
        full_ref_loss = ref_loss + ref_loss_reg
        print(f"ref full loss (with reg): {full_ref_loss}")  # with regularization

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
            self.liveloss.update({'data term': loss.item(), 'reg': reg.item(), 'full_loss': full_loss.item()})
            self.liveloss.send()

    def _create_optimizer(self):
        lr = self.params['lr']
        solver_type = self.params.get("solver_type", None)
        model_params = self.composite_nn.parameters()
        if solver_type == 'gd':
            optimizer = optim.SGD(params=model_params, lr=lr)
        elif solver_type == 'lbfgs':
            optimizer = optim.LBFGS(params=model_params, lr=lr, max_iter=5,
                                    max_eval=None,
                                    tolerance_grad=1e-07,
                                    tolerance_change=1e-09,
                                    history_size=10,
                                    line_search_fn=None)
        elif solver_type == 'adam':
            weight_decay = self.params['weight_decay']
            optimizer = optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
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

        return A, A_sub, G, G_sub

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

    def __embedding_sub(self, G: nx.graph, G_sub: nx.graph, dtype):
        A, A_sub, G, G_sub = self.__pre_process_graphs(G, G_sub)
        device = self.params['device']
        A = A.to(device=device)
        A_sub = A_sub.to(device=device)

        embeddings_sub = self.composite_nn.embed(A=A_sub.detach().type(dtype),
                                                 w=uniform_dist(A_sub.shape[0]).detach().to(device=self.params['device']))

        return A, A_sub, G, G_sub, embeddings_sub

    def get_loss_for_graph_and_subgraph(self, G: nx.graph, G_sub: nx.graph, dtype=torch.double):
        A, A_sub, G, G_sub, embeddings_sub = self.__embedding_sub(G, G_sub, dtype)
        loss, reg = self.get_composite_loss_terms(A, embeddings_sub)
        return loss + reg

    def solve(self, G: nx.graph, G_sub: nx.graph, dtype=torch.double):
        A, A_sub, G, G_sub, embeddings_sub = self.__embedding_sub(G, G_sub, dtype)

        self.composite_nn.train() # Set the model to training mode
        optimizer = self._create_optimizer()

        for iteration in range(self.params["maxiter"]):  # TODO: add stopping condition
            def closure():
                loss, reg = self.get_composite_loss_terms(A, embeddings_sub)
                full_loss = loss + reg

                optimizer.zero_grad()
                full_loss.backward()

                self.__log_loss(iteration, loss, reg)

                return full_loss

            optimizer.step(closure)

        w_star = self.composite_nn.node_classifier_network(A=A, params=self.params).detach().cpu().numpy()
        return w_star

    def set_initial_params_based_on_previous_optimum(self, w_star):
        # binarized_w_star = IndicatorDistributionBinarizer.binarize(processed_G, w_star, self.params,
        #                                  binarization_type)
        # w_th = torch.tensor(list(binarized_w_star.values()), device=device)

        device = self.params['device']
        x0 = w_star-np.min(w_star)
        x0 = torch.tensor(x0 / x0.sum(), device=device)
        self.composite_nn.node_classifier_network.init_params(default_weights=x0)
        return x0