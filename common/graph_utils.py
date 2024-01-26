import itertools
import random
import networkx as nx


def relabel_graph_nodes_by_contiguous_order(g: nx.Graph, copy):
    nodes_map = dict(zip(list(g.nodes), [i for i in range(len(g))]))
    graph = nx.relabel_nodes(g, nodes_map, copy=copy)
    return graph


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