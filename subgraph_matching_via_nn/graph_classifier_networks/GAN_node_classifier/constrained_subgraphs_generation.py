import multiprocessing

from common.graph_utils import SubGraphGenerator
from common.logger import TimeLogging
from subgraph_matching_via_nn.data.annotated_graph import AnnotatedGraph
from subgraph_matching_via_nn.data.data_loaders import load_graph
from subgraph_matching_via_nn.data.paths import DATA_PATH

if __name__ == '__main__':

# use the k subgraphs generation utils methods (only keep specified edges number k[node]-subgraphs)

    samples_n = 1_000 # generated population max size

    graph_n_nodes = 20
    subgraph_n_nodes = 7

    COMP1_FULL_path = 'comp1_2\\full_graph.p'
    COMP1_SUB0_path = 'comp1_2\\subgraph0.p'

    # load graph instance
    loader_params = {'graph_size': graph_n_nodes,
                     'subgraph_size': subgraph_n_nodes,
                     'data_path': DATA_PATH,
                     'g_full_path': COMP1_FULL_path,
                     'g_sub_path': COMP1_SUB0_path}

    sub_graph = \
        load_graph(type='random',
                   loader_params=loader_params)  # type = 'random', 'example', 'subcircuit'

    subgraph_n_nodes = len(sub_graph.G_sub.nodes)
    subgraph_n_edges = len(sub_graph.G_sub.edges)

    print(len(sub_graph.G.nodes))
    print(subgraph_n_nodes)
    print(subgraph_n_edges)

    # generate k subgraph
    is_parallel = True
    multiprocessing.set_start_method("spawn")

    G_sub = sub_graph.G_sub
    k = len(G_sub)
    n = len(sub_graph.G)
    G_perturbed = sub_graph.G.copy()
    # candidate_nodes_to_remove_from_full_graph = list(set(sub_graph.G.nodes).difference(G_sub.nodes))

    # n_nodes_to_remove_from_full_graph = len(candidate_nodes_to_remove_from_full_graph) // 2
    # G_perturbed.remove_nodes_from(candidate_nodes_to_remove_from_full_graph[:n_nodes_to_remove_from_full_graph])
    #
    # remove_isolated_nodes_from_graph(G_perturbed)
    #
    # print(
    #     f"full graph has {n} nodes, subgraph has {k} nodes, removing {n - len(G_perturbed)} non subgraph nodes from full graph")

    source_graph = G_perturbed
    print("starting generating subgraphs")
    curr_time = TimeLogging.log_time(None, "start generate_k_subgraphs")

    k_subgraphs, k_subgraphs_original_nodes = SubGraphGenerator.generate_k_subgraphs(source_graph, k=k, is_parallel=is_parallel)
    # k_subgraph_annotated_graphs = [AnnotatedGraph(g, label=i) for i, g in enumerate(k_subgraphs)]

    print(len(k_subgraphs))

    curr_time = TimeLogging.log_time(curr_time, "end generate_k_subgraphs")

    # filter subgraphs by constraint
    filtered_k_subgraphs = [k_subgraph for k_subgraph in k_subgraphs if len(k_subgraph.edges) == subgraph_n_edges]

    print(len(filtered_k_subgraphs))
