import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from subgraph_matching_via_nn.data.data_loaders import load_graph
from subgraph_matching_via_nn.data.paths import *
from subgraph_matching_via_nn.evaluation.mask_evaluation import compute_relative_hausdorff_distance_to_target_subgraph


# def generate_example_graph(base_node_id):
#     # Example graph
#     G = nx.Graph()
#     graph_edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
#     for u,v in graph_edges:
#         u += base_node_id
#         v += base_node_id
#         G.add_edges_from([(u, v)])
#     # Target subgraph
#     target_subgraph = {base_node_id + _id for _id in {1, 2, 3, 4}}
#     # Actual subgraph
#     actual_subgraph = {base_node_id + _id for _id in {3, 4, 5, 6}}
#     # Compute nodes in both subgraphs
#     overlap_nodes = target_subgraph.intersection(actual_subgraph)
#     # Compute nodes only in the target subgraph
#     target_only_nodes = target_subgraph - overlap_nodes
#     # Compute nodes only in the actual subgraph
#     actual_only_nodes = actual_subgraph - overlap_nodes
#
#     return G, target_subgraph, actual_subgraph, target_only_nodes, actual_only_nodes, overlap_nodes




# Function to compute subgraph localization accuracy score


loader_params = {'data_path': DATA_PATH,
                 'g_full_path': 'comp1_4\\full_graph.p',
                 'g_sub_path': 'comp1_4\\subgraph0.p'}

sub_graph = \
    load_graph(type='subcircuit',
               loader_params=loader_params)  # type = 'random', 'example', 'subcircuit'

original_G = sub_graph.G
G = original_G
G_sub = sub_graph.G_sub
target_subgraph = G_sub.nodes
actual_subgraph = set(sorted(random.sample(G.nodes, len(G_sub)), reverse=False))
graph_edges = list(G.edges())

# target_subgraph = {1, 2, 3, 4}
# actual_subgraph = {3, 4, 5, 6}
# graph_edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]

print(target_subgraph)
print(actual_subgraph)
print(graph_edges)

def generate_example_graph(base_node_id, actual_subgraph, target_subgraph):
    target_subgraph = {base_node_id + node for node in target_subgraph}
    actual_subgraph = {base_node_id + node for node in actual_subgraph}

    G = nx.Graph()

    for u,v in graph_edges:
        u += base_node_id
        v += base_node_id
        G.add_edges_from([(u, v)])

    overlap_nodes = target_subgraph.intersection(actual_subgraph)
    target_only_nodes = target_subgraph - overlap_nodes
    actual_only_nodes = actual_subgraph - overlap_nodes

    return G, target_subgraph, actual_subgraph, target_only_nodes, actual_only_nodes, overlap_nodes

G1, target_subgraph1, actual_subgraph1, target_only_nodes1, actual_only_nodes1, overlap_nodes1 = \
    generate_example_graph(base_node_id=0, actual_subgraph=actual_subgraph, target_subgraph=target_subgraph)
G2, target_subgraph2, actual_subgraph2, target_only_nodes2, actual_only_nodes2, overlap_nodes2 =\
    generate_example_graph(base_node_id=len(G1), actual_subgraph=actual_subgraph, target_subgraph=target_subgraph)

alpha, shortest_path_distances_from_target = compute_relative_hausdorff_distance_to_target_subgraph(G2, target_subgraph2, actual_subgraph2)
_, shortest_path_distances_to_target = compute_relative_hausdorff_distance_to_target_subgraph(G1, actual_subgraph1, target_subgraph1)

target_only_nodes = set(list(target_only_nodes1) + list(target_only_nodes2))
actual_only_nodes = set(list(actual_only_nodes1) + list(actual_only_nodes2))
overlap_nodes = set(list(overlap_nodes1) + list(overlap_nodes2))

G = nx.union(G1, G2)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
pos = nx.bipartite_layout(G, G1.nodes)

labels = {g_node:g_node - len(G2) if g_node >= len(G2) else g_node for g_node in G.nodes}

nx.draw(G, pos=pos, with_labels=True, nodelist=target_only_nodes, node_color='red', node_size=500, label='GT Subgraph Nodes', labels=labels)
nx.draw(G, pos=pos, with_labels=True, nodelist=overlap_nodes, node_color='yellow', node_size=500, label='Overlap Nodes', labels=labels)
nx.draw(G, pos=pos, with_labels=True, nodelist=actual_only_nodes, node_color='green', node_size=500, label='Localized Subgraph Nodes', labels=labels)

for actual_node in actual_subgraph2:
    distances_map = shortest_path_distances_from_target[actual_node]
    min_distance = len(G1)
    for target_node in target_subgraph2:
        distance = distances_map.get(target_node, None)
        min_distance = min(min_distance, distance)
    assert min_distance < len(G1)

    distance = min_distance
    if distance > 0:
        target_node_to_plot = target_node - len(G2)
        actual_node_to_plot = actual_node

        width = 1 + distance / alpha
        plt.plot([pos[target_node_to_plot][0], pos[actual_node_to_plot][0]],
                 [pos[target_node_to_plot][1], pos[actual_node_to_plot][1]],
                 linestyle='dotted', color='blue', linewidth=width)
        plt.text((1*pos[target_node_to_plot][0] + 2*pos[actual_node_to_plot][0]) / 3,
                 (1*pos[target_node_to_plot][1] + 2*pos[actual_node_to_plot][1]) / 3,
                 str(distance), color='black', fontsize=13)

for target_node in target_subgraph1:
    distances_map = shortest_path_distances_to_target[target_node]
    min_distance = len(G1)
    for actual_node in actual_subgraph1:
        distance = distances_map.get(actual_node, None)
        min_distance = min(min_distance, distance)
    assert min_distance < len(G1)

    distance = min_distance
    if distance > 0:
        actual_node_to_plot = actual_node + len(G2)

        width = 1 + distance / alpha
        line = plt.plot([pos[actual_node_to_plot][0], pos[target_node][0]],
                 [pos[actual_node_to_plot][1], pos[target_node][1]],
                 linestyle='dotted', color='gray', linewidth=width)
        # Plot the arrowhead using plt.annotate
        plt.text((1*pos[actual_node_to_plot][0] + 2*pos[target_node][0]) / 3,
                 (1*pos[actual_node_to_plot][1] + 2*pos[target_node][1]) / 3,
                 str(distance), color='black', fontsize=13)


# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#            fancybox=True, shadow=True, ncol=3)


# Draw nodes with specific colors and store the handles
target_patch = mpatches.Patch(color='red', label='exclusive GT Subgraph Nodes')
overlap_patch = mpatches.Patch(color='yellow', label='Overlap Nodes')
actual_patch = mpatches.Patch(color='green', label='exclusive Localized Subgraph Nodes')

plt.legend(handles=[target_patch, overlap_patch, actual_patch], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)


plt.tight_layout()
plt.show()



_, target_subgraph3, actual_subgraph3, target_only_nodes3, actual_only_nodes3, overlap_nodes3 = \
    generate_example_graph(base_node_id=0, actual_subgraph=actual_subgraph, target_subgraph=target_subgraph)
# Create a new figure
plt.figure(figsize=(12, 6))

# Create two subplots, one for the left version and one for the right version
plt.subplot(1, 2, 1)


labels = {g_node: g_node for g_node in original_G.nodes}

pos = nx.spring_layout(original_G, seed=42)
nx.draw(original_G, pos=pos, with_labels=True, nodelist=target_only_nodes3, node_color='red', node_size=500, label='GT Subgraph Nodes', labels=labels)
nx.draw(original_G, pos=pos, with_labels=True, nodelist=overlap_nodes3, node_color='yellow', node_size=500, label='Overlap Nodes', labels=labels)
nx.draw(original_G, pos=pos, with_labels=True, nodelist=actual_only_nodes3, node_color='green', node_size=500, label='Localized Subgraph Nodes', labels=labels)

# Adding edge colors
edge_colors = []
target_edges = list(G_sub.edges())
for edge in original_G.edges():
    is_target_edge = (edge in target_edges) and (edge[0] in target_subgraph3 and edge[1] in target_subgraph3)
    is_actual_edge = (edge[0] in actual_subgraph3 and edge[1] in actual_subgraph3)
    is_overlap_edge = (is_target_edge and is_actual_edge)

    if is_overlap_edge:
        edge_colors.append('yellow')  # Color for edges within Overlap Nodes
    elif is_target_edge:
        edge_colors.append('red')  # Color for edges within GT Subgraph Nodes
    elif is_actual_edge:
        edge_colors.append('green')  # Color for edges within Overlap Nodes
    else:
        edge_colors.append('black')  # Default edge color

nx.draw_networkx_edges(original_G, pos=pos, edge_color=edge_colors)

# Draw nodes with specific colors and store the handles
target_patch = mpatches.Patch(color='red', label='exclusive GT Subgraph Nodes/Edges')
overlap_patch = mpatches.Patch(color='yellow', label='Overlap Nodes/Edges')
actual_patch = mpatches.Patch(color='green', label='exclusive Localized Subgraph Nodes/Edges')
general_patch = mpatches.Patch(color='black', label='exclusive non-GT Subgraph Nodes/Edges')

plt.legend(handles=[target_patch, overlap_patch, actual_patch, general_patch], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)



plt.tight_layout()
plt.show()
#
# edge_colors_dbg = []
# for edge in original_G.edges():
#     is_target_edge = (edge in target_edges) and (edge[0] in target_only_nodes3 and edge[1] in target_only_nodes3)
#     is_actual_edge = (edge[0] in actual_only_nodes3 and edge[1] in actual_only_nodes3)
#     is_overlap_edge = (is_target_edge and is_actual_edge)
#
#     if is_overlap_edge:
#         edge_colors_dbg.append('yellow')  # Color for edges within Overlap Nodes
#     elif is_target_edge:
#         edge_colors_dbg.append('red')  # Color for edges within GT Subgraph Nodes
#     elif is_actual_edge:
#         edge_colors_dbg.append('green')  # Color for edges within Overlap Nodes
#     else:
#         edge_colors_dbg.append('black')  # Default edge color