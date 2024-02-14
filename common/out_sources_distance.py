from torch import nn
import torch
import numpy as np
from torch.autograd import Variable as V

REALLY_BIG_NUM = float("inf")


def build_param_graph(G, device):
    parameter_graph = []
    parameters = []
    for i in range(len(G)):
        edges = []
        parameter_graph.append(edges)
        for j in range(len(G)):
            if G[i][j] != 0:
                d = G[i][j]
                param = nn.Parameter(torch.ones(1, device=device))
                parameters.append(param)
                edges.append((j, param, d))
    return parameter_graph, parameters


def torch_dijsktra(gr, device):
    graph_dim = len(gr)
    graph_copy = torch.ones((graph_dim, graph_dim), device=device) * REALLY_BIG_NUM

    distances = np.empty(graph_dim, dtype=np.float32)
    for i, edges in enumerate(gr):
        distances.fill(REALLY_BIG_NUM)
        distances[i] = 0
        torch_dist = graph_copy[i]
        torch_dist[i] = V(torch.zeros(1, device=device))
        for _ in range(graph_dim):
            v = distances.argmin()
            v_dist = torch_dist[v]
            distances[v] = np.inf  # won't be selected by argmin , i.e. removed from the pool
            for neighbor, d, min_d in gr[v]:
                new_d = v_dist + d.clamp(min=0, max=1) * min_d
                existing_d = torch_dist[neighbor]
                if existing_d is None or (new_d < existing_d):
                    torch_dist[neighbor] = new_d
                    distances[neighbor] = new_d.detach().cpu().numpy()[0]

    return graph_copy


def get_out_sources_max_distance(target_graph, full_graph_adj, source_graphs, graph_node_to_adj_node_map, device):
    parameter_graph, parameters = build_param_graph(full_graph_adj, device)

    dijkstra_distances = torch_dijsktra(parameter_graph, device)
    target_graph_nodes_as_sources_dijkstra_distances = torch.stack([dijkstra_distances[graph_node_to_adj_node_map[node]] for node in target_graph.nodes]).T
    shortest_paths_to_target_graph_nodes = torch.min(target_graph_nodes_as_sources_dijkstra_distances, dim=1).values

    # access adj_matrix via node -> adj_node map passed as a parameter
    subgraphs_mean_distances = [
        torch.max(torch.stack([shortest_paths_to_target_graph_nodes[graph_node_to_adj_node_map[node]] for node in subgraph.nodes]))
        for subgraph in source_graphs]

    return subgraphs_mean_distances


if __name__ == "__main__":
    from torch import optim
    device = 'cuda'
    parameter_graph, parameters = build_param_graph(sub_graph.A_sub
                                                    , device)

    optimizer = optim.SGD(parameters,
                          lr=0.01, momentum=0.5)

    for epoch in range(100):
        dijkstra_distances = torch_dijsktra(parameter_graph, device)
        loss = torch.stack([e for lst in dijkstra_distances for e in lst]).sum()

        print(loss)
        epoch_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()