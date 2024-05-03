import networkx as nx
import pulp
import torch
from pulp import PULP_CBC_CMD

from subgraph_matching_via_nn.graph_classifier_networks.classification_layer.classification_layer import \
    TopkSoftmaxClassificationLayer
from subgraph_matching_via_nn.graph_classifier_networks.node_classifier_networks import IdentityNodeClassifierNetwork
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


def solve_maximum_weight_subgraph(weights, adjacency_matrix, requested_num_nodes, requested_num_edges):
    # Create a binary integer programming problem
    problem = pulp.LpProblem("Maximum_Weight_Subgraph", pulp.LpMaximize)

    # Variables
    num_nodes = len(weights)
    x = [pulp.LpVariable(f"x{i}", cat=pulp.LpBinary) for i in range(num_nodes)]
    y = {(i, j): pulp.LpVariable(f"y{i}_{j}", cat=pulp.LpBinary) for i in
         range(num_nodes) for j in range(i + 1, num_nodes)}

    # Objective function
    problem += pulp.lpSum(weights[i] * x[i] for i in range(num_nodes))

    # Constraints
    problem += pulp.lpSum(x) == requested_num_nodes  # Node selection constraint
    problem += pulp.lpSum(y.values()) == requested_num_edges  # Edge selection constraint

    # Connectivity constraints
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i][j] == 0:
                problem += y[(i, j)] == 0  # If there is no edge, it cannot be selected
            else:
                problem += y[(i, j)] <= x[i]  # Relationship between y_ij and x_i
                problem += y[(i, j)] <= x[j]

    # Solve the problem
    #pulp.GUROBI(msg=0).solve(problem)

    problem.solve(PULP_CBC_CMD(msg=0))

    # Extract the solution
    selected_nodes = [i for i in range(num_nodes) if pulp.value(x[i]) == 1]
    selected_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)
                      if pulp.value(y[(i, j)]) == 1]

    # Return the selected nodes and edges
    return selected_nodes, selected_edges