import itertools
import math
import random
from multiprocessing import Pool
from os import cpu_count
import sys
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm
from common.logger import TimeLogging
from common.parallel_computation import tqdm_joblib


def relabel_graph_nodes_by_contiguous_order(g: nx.Graph, copy):
    nodes_map = dict(zip(list(g.nodes), [i for i in range(len(g))]))
    graph = nx.relabel_nodes(g, nodes_map, copy=copy)
    return graph


def fill_subgraph_missing_nodes(subgraph, start_graph):
    subgraph.add_nodes_from(list(start_graph.nodes))


class GED_graph_generator:
    def __init__(self, reference_graph: nx.Graph, ged_dist: int):
        self.reference_graph = reference_graph
        self.ged_dist = ged_dist

        assert ged_dist > 0

    def generate(self):
        # assumption: there are no self loops, and no duplicate edges of opposite directions
        # don't generate same graph twice, generate in random
        # only produce GED exmaples which are for sure not pairs of isomoprhic graphs (e.g. by adding an edge and removing another edge)

        # generate all GED-1 operations
        ged_1_add_operations_list = []
        ged_1_remove_operations_list = []
        sorted_nodes = sorted(self.reference_graph.nodes, reverse=False)
        for i, node_i in enumerate(sorted_nodes):
            if i == len(sorted_nodes) - 1:
                break

            for node_j in sorted_nodes[i+1:]:
                edge = (node_i, node_j)
                if self.reference_graph.has_edge(*edge):
                    ged_1_remove_operations_list.append(("remove", edge))
                else:
                    ged_1_add_operations_list.append(("add", edge))

        # generate homogenic operation ged_dist operations lists
        ged_dist_add_operations_list = itertools.combinations(ged_1_add_operations_list, self.ged_dist)
        ged_dist_remove_operations_list = itertools.combinations(ged_1_remove_operations_list, self.ged_dist)

        def generate_operations_series_list(reference_graph, ged_dist_operations_list):
            copy_graph = reference_graph.copy()

            for ged_operation, edge in ged_dist_operations_list:
                if ged_operation == "remove":
                    copy_graph.remove_edge(*edge)
                else:
                    copy_graph.add_edge(*edge)
            return copy_graph

        is_choosing_next_list_randomly = True
        while True:
            if is_choosing_next_list_randomly:
                next_list_choise = random.randint(0, 1)

            try:
                if next_list_choise == 0:
                    ged_dist_operations_list = next(ged_dist_add_operations_list)
                else:
                    ged_dist_operations_list = next(ged_dist_remove_operations_list)
            except StopIteration:
                ged_dist_operations_list = None

            if ged_dist_operations_list is None:
                if not is_choosing_next_list_randomly:
                    break
                next_list_choise = 1 - next_list_choise
                is_choosing_next_list_randomly = False
                continue

            copy_graph = generate_operations_series_list(self.reference_graph, ged_dist_operations_list)
            yield copy_graph
        print("finished generation")


class SubGraphGenerator:

    @staticmethod
    def generate_subgraph(graph, connected_subgraphs_nodes_list):
        subgraph = graph.subgraph(connected_subgraphs_nodes_list).copy()
        fill_subgraph_missing_nodes(subgraph, graph)

        return subgraph

    @staticmethod
    def generate_subgraph_for_sublists_of_nodes(graph, connected_subgraphs_nodes_lists):
        res = [SubGraphGenerator.generate_subgraph(graph, connected_subgraphs_nodes_list)
                for connected_subgraphs_nodes_list in connected_subgraphs_nodes_lists]
        return res

    # https://stackoverflow.com/questions/75727217/fast-way-to-find-all-connected-subgraphs-of-given-size-in-python
    @staticmethod
    def all_connected_subgraphs(g, m):
        found_subgraphs = []
        n = len(g.nodes)
        adj_sets_map = {i:set() for i in g.nodes}
        for (i, j) in g.edges:
            adj_sets_map[i].add(j)
            adj_sets_map[j].add(i)

        def _recurse(t, possible, excluded, found_subgraphs=[]):
            if len(t) == m:
                found_subgraphs.append(t)
                return
            else:
                excluded = set(excluded)
                for i in possible:
                    if i not in excluded:
                        new_t = (*t, i)
                        new_possible = possible | set(g[i].keys())
                        excluded.add(i)
                        _recurse(new_t, new_possible, excluded, found_subgraphs)

        excluded = set()
        for node_i, possible in adj_sets_map.items():
            excluded.add(node_i)
            _recurse((node_i,), possible, excluded, found_subgraphs)
        return found_subgraphs

    @staticmethod
    def generate_k_subgraphs_for_chunk(graph, chunk_index, chunk):
        curr_time = TimeLogging.log_time(None, "enter generate_k_subgraphs_for_chunk")

        res = SubGraphGenerator.generate_subgraph_for_sublists_of_nodes(graph, chunk)

        curr_time = TimeLogging.log_time(curr_time, f"Chunk #{chunk_index} finished, chunk size={len(chunk)}")
        sys.stdout.flush()
        return res

    @staticmethod
    def generate_k_subgraphs(graph, k, is_parallel=True):
        curr_time = TimeLogging.log_time(None, "enter generate_k_subgraphs")
        cpu_num = int(cpu_count())
        all_connected_subgraphs_nodes_lists = SubGraphGenerator.all_connected_subgraphs(graph, k)
        n = len(all_connected_subgraphs_nodes_lists)

        if is_parallel:
            chunks_amount = cpu_num
        else:
            chunks_amount = 1

        chunk_size = int(math.ceil(n / chunks_amount))
        curr_time = TimeLogging.log_time(curr_time, f"finished all_connected_subgraphs (total of {n} graphs)")
        
        chunks = [all_connected_subgraphs_nodes_lists[i*chunk_size: min(i*chunk_size + chunk_size, n)] for i in range(chunks_amount)]

        if is_parallel:
            # create and configure the process pool
            with Pool(processes=cpu_num) as pool:
                # execute tasks in order
                subgraph_lists_list = pool.starmap(SubGraphGenerator.generate_k_subgraphs_for_chunk,
                                                   zip(itertools.repeat(graph), range(chunks_amount), chunks))
            subgraphs_list = [e for lst in subgraph_lists_list for e in lst]
        else:
            subgraphs_list = SubGraphGenerator.generate_k_subgraphs_for_chunk(graph, 1, chunks[0])

        curr_time = TimeLogging.log_time(curr_time, "finished generating subgraphs")

        return subgraphs_list
