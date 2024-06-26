from abc import abstractmethod, ABC
from typing import Callable, List, Dict, Any
import networkx as nx
import numpy as np
import torch
from overrides import override
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT
from subgraph_matching_via_nn.mask_binarization.LP_binarization import solve_maximum_weight_subgraph
from torch.optim.sgd import SGD


class BaseFrankWolfeOptimizer(SGD, ABC):

    def __init__(self, params: ParamsT, defaults: Dict[str, Any], processed_graph: nx.Graph,
                 gradient_average_iterations_amount: int):
        ABC.__init__(self)
        SGD.__init__(self, params=params, **defaults)
        self.processed_graph = processed_graph
        self.gradient_average_iterations_amount = gradient_average_iterations_amount

    def _step_via_sgd(self, closure):
        if closure is not None:
            gradient_average_iterations_amount = self.gradient_average_iterations_amount
            if gradient_average_iterations_amount != 1:
                # apply closure multiple times
                for closure_iter in range(gradient_average_iterations_amount):
                    # this performs a single gradient step
                    super().step(
                        closure=lambda:
                        closure(inner_optimizer_iteration=closure_iter, is_log=False, is_calc_grad=True)
                    )

    @abstractmethod
    def _get_grads_as_dict(self) -> Dict:
        raise NotImplementedError()

    @abstractmethod
    def _binarize_mask(self, grads_dict) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _update_model_params_to_match_mask(self, w_mask, closure):
        raise NotImplementedError()

    def step(self, closure=None):
        # allow for averages gradient iterations, not just immediate FW
        self._step_via_sgd(closure)

        # get grads
        grads_dict = self._get_grads_as_dict()

        # binarization
        selected_nodes = self._binarize_mask(grads_dict)
        w_th = np.zeros([len(self.processed_graph.nodes()), 1])
        w_th[selected_nodes] = 1.0
        w_th = w_th / w_th.sum()

        # change param values according to resulting mask
        self._update_model_params_to_match_mask(w_th, closure)
        print("finished _update_model_params_to_match_mask")

        #log loss
        _ = closure(inner_optimizer_iteration=None, is_log=True, is_calc_grad=False)


class LPFrankWolfeOptimizer(BaseFrankWolfeOptimizer):
    def __init__(self, params: ParamsT, is_working_on_node_mask: bool, num_nodes: int, num_edges: int,
                 original_graph: nx.Graph, processed_graph: nx.Graph,
                 acquire_mask_gradients_lambda: Callable[[], List[float]],
                 gradient_average_iterations_amount: int, kwargs):
        super().__init__(params=params, defaults=kwargs, processed_graph=processed_graph,
                         gradient_average_iterations_amount=gradient_average_iterations_amount)

        self.is_working_on_node_mask = is_working_on_node_mask
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.original_graph = original_graph
        self.acquire_mask_gradients_lambda = acquire_mask_gradients_lambda

    @override
    def _get_grads_as_dict(self) -> Dict:
        # fetch grad: the loss parameters are not the ones we are after, but the actual output mask!
        grad_list = self.acquire_mask_gradients_lambda()
        grad_list = [grad if grad is not None else 0 for grad in grad_list]
        # print(f"_get_grads_as_dict:{grad_list}")
        # convert grad_list to a dict (use the mask keys order)
        grad_list = dict(zip(self.processed_graph.nodes(), grad_list))

        return grad_list

    @override
    def _binarize_mask(self, grads_dict) -> np.ndarray:
        # apply LP solver on the grad values

        num_nodes = self.num_nodes
        num_edges = self.num_edges

        grads_dict = {k: -v for k, v in grads_dict.items()}
        selected_nodes, selected_edges = solve_maximum_weight_subgraph(grads_dict, self.original_graph, num_nodes, num_edges)
        # print(f'requested: n_nodes = {num_nodes}, n_edges : {num_edges}')
        # print(f'found: n_nodes = {len(selected_nodes)}, n_edges : {len(selected_edges)}')
        # print(selected_edges)

        # convert resulting mask W to the format the processed graph is working with (in terms of line graph format)
        if self.is_working_on_node_mask:
            pass
        else:
            # if working on a line graph, convert the result edges mask to the node mask we are working on
            selected_nodes = selected_edges

        return selected_nodes

    @abstractmethod
    def _update_model_params_to_match_mask(self, w_mask, closure):
        raise NotImplementedError()


class IdentityNodeClassifierLPFrankWolfeOptimizer(LPFrankWolfeOptimizer):
    def __init__(self, params: ParamsT, is_working_on_node_mask: bool, num_nodes: int, num_edges: int,
                 original_graph: nx.Graph, processed_graph: nx.Graph,
                 acquire_mask_gradients_lambda: Callable[[], List[float]],
                 gradient_average_iterations_amount: int, **kwargs):
        super().__init__(params=params, is_working_on_node_mask=is_working_on_node_mask, num_nodes=num_nodes,
                         num_edges=num_edges, original_graph=original_graph, processed_graph=processed_graph,
                         acquire_mask_gradients_lambda=acquire_mask_gradients_lambda,
                         gradient_average_iterations_amount=gradient_average_iterations_amount, kwargs=kwargs)

    @override
    def _update_model_params_to_match_mask(self, w_mask, closure):
        # calculate grad
        _ = closure(is_log=False, is_calc_grad=True)

        mask_tensor = self.param_groups[0]['params'][0]
        mask_grad = mask_tensor.grad
        # print(f"update model params grad: {mask_grad.reshape(-1)}")
        for i, param_grad in enumerate(mask_grad):
            if param_grad is None:
                continue
            mask_tensor[i].data.fill_(w_mask[i][0])


class DeepNodeClassifierLPFrankWolfeOptimizer(LPFrankWolfeOptimizer):

    INVERSE_MASK_SGD_ITER_AMOUNT = 20

    def __init__(self, params: ParamsT, is_working_on_node_mask: bool, num_nodes: int, num_edges: int,
                 original_graph: nx.Graph, processed_graph: nx.Graph,
                 acquire_mask_gradients_lambda: Callable[[], List[float]],
                 get_output_mask: Callable[[], torch.Tensor],
                 gradient_average_iterations_amount: int, **kwargs):
        super().__init__(params=params, is_working_on_node_mask=is_working_on_node_mask, num_nodes=num_nodes,
                         num_edges=num_edges, original_graph=original_graph, processed_graph=processed_graph,
                         acquire_mask_gradients_lambda=acquire_mask_gradients_lambda,
                         gradient_average_iterations_amount=gradient_average_iterations_amount, kwargs=kwargs)
        self.get_output_mask = get_output_mask

    @override
    def _update_model_params_to_match_mask(self, w_mask, closure):
        # inverse the mask to the actual params

        # Define the loss function
        criterion = torch.nn.MSELoss()

        # Define the optimizer
        model_params = sum([group['params'] for group in self.param_groups], [])
        optimizer = torch.optim.SGD(model_params, lr=0.01)

        some_param = model_params[0]
        target_mask = torch.tensor(w_mask, device=some_param.device, dtype=some_param.dtype)

        # Training loop
        num_epochs = DeepNodeClassifierLPFrankWolfeOptimizer.INVERSE_MASK_SGD_ITER_AMOUNT
        for epoch in range(num_epochs):
            # Forward pass
            y_pred = self.get_output_mask()

            # Compute the loss
            loss = criterion(y_pred, target_mask)

            # Zero gradients, backward pass, and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # Print progress
            # if (epoch + 1) % 10 == 0:
            #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')