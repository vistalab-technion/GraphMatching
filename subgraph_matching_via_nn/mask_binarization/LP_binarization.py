import networkx as nx
import pulp
from pulp import PULP_CBC_CMD


class LPBinarizationProblemType:
    Continuous=1,
    Binary=2,


def solve_maximum_weight_subgraph(weights : dict,
                                  graph: nx.Graph,
                                  requested_num_nodes,
                                  requested_num_edges,
                                  problem_type: LPBinarizationProblemType = LPBinarizationProblemType.Binary):
    """
    Solves a maximum weighted subgraph problem. If 'weights' is a node weights vector (i.e.,
    it of size [nodes x 1]), then it solves:

    max_{x,y} weights.T @ x

    s.t.

    sum(x) = requested_num_nodes
    sum(y) = requested_num_edges
    y[i,j]<= x[i] & y[i,j]<=x[j] if A[i,j]==1, else y[i,j]=0

    i.e., it finds the best weighted subset of nodes that correspond to a subgraph with
    requested_num_nodes and requested_num_edges.

    If 'weights' is an edge weight vector (i.e., it of size [edges x 1]), then the objective is

    max_{x,y} weights.T @ y

    s.t same constraints

    Example usage
    weights = [3, 4, 5, 2, 1]
    adjacency_matrix = [
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ]
    k = 3  # Number of nodes in the subgraph
    l = 3  # Number of edges in the subgraph

    selected_nodes, selected_edges = solve_maximum_weight_subgraph(weights,
                                                                   adjacency_matrix,
                                                                   k,
                                                                   l)
    print("Selected nodes:", selected_nodes)
    print("Selected edges:", selected_edges)


    """
    # Create a binary integer programming problem
    problem = pulp.LpProblem("Maximum_Weight_Subgraph", pulp.LpMaximize)

    # Variables
    adjacency_matrix = (nx.adjacency_matrix(graph)).toarray()
    num_nodes = len(graph.nodes)
    num_edges = len(graph.edges)

    category: LPBinarizationProblemType

    if problem_type == LPBinarizationProblemType.Binary:
        category = pulp.LpBinary
    elif problem_type == LPBinarizationProblemType.Continuous:
        category = pulp.LpContinuous
    else:
        raise NotImplementedError(f"LP problem type not supported: {problem_type}")

    x = [pulp.LpVariable(f"x{i}", cat=category) for i in range(num_nodes)]
    y = {(i, j): pulp.LpVariable(f"y{i}_{j}", cat=category) for i in
         range(num_nodes) for j in range(i + 1, num_nodes)}

    # Objective function
    if len(weights) == num_nodes:
        problem += pulp.lpSum(weights[node] * x[node] for node in graph.nodes)
    elif len(weights) == num_edges:
        problem += pulp.lpSum(
            weights[edge] * y[edge] for edge in graph.edges())

    # Constraints
    problem += pulp.lpSum(x) == requested_num_nodes  # Node selection constraint
    problem += pulp.lpSum(
        y.values()) == requested_num_edges  # Edge selection constraint

    # Connectivity constraints
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i][j] == 0:
                problem += y[(i, j)] == 0  # If there is no edge, it cannot be selected
            else:
                problem += y[(i, j)] <= x[i]  # Relationship between y_ij and x_i
                problem += y[(i, j)] <= x[j]

    # Solve the problem
    # pulp.GUROBI(msg=0).solve(problem)

    problem.solve(PULP_CBC_CMD(msg=0))

    # Extract the solution
    if problem_type == LPBinarizationProblemType.Binary:
        selected_nodes_dict = {i: 1 for i in range(num_nodes) if pulp.value(x[i]) == 1}
        selected_edges_dict = {(i, j): 1 for i in range(num_nodes) for j in range(i + 1, num_nodes)
                          if pulp.value(y[(i, j)]) == 1}
    else:
        selected_nodes_dict = {i: pulp.value(x[i]) for i in range(num_nodes)}
        # no need to take into account edges which don't exist in the first place!
        selected_edges_dict = {(i, j): pulp.value(y[(i, j)]) for i in range(num_nodes) for j in range(i + 1, num_nodes)
                          if adjacency_matrix[i][j] != 0}

    # Return the selected nodes and edges mapping
    return selected_nodes_dict, selected_edges_dict
