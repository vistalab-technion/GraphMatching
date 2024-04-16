from __future__ import print_function
# print(__doc__)

import numpy as np
import networkx as nx
from grakel import SubgraphMatching

from grakel.utils import graph_from_networkx

# Creates a list of two simple graphs
G1 = nx.Graph()
G1.add_nodes_from([0,1,2])
G1.add_edges_from([(0,1), (1,2)])

G2 = nx.Graph()
G2.add_nodes_from([0,1,2])
G2.add_edges_from([(0,1), (0,2), (1,2)])

G_nx = [G1, G2]

# Transforms list of NetworkX graphs into a list of GraKeL graphs
G = graph_from_networkx(G_nx)
print("1 - Simple graphs transformed\n")


# Creates a list of two node-labeled graphs
G1 = nx.Graph()
G1.add_nodes_from([0,1,2])
G1.add_edges_from([(0,1), (1,2)])
nx.set_node_attributes(G1, {0:'a', 1:'b', 2:'a'}, 'label')

G2 = nx.Graph()
G2.add_nodes_from([0,1,2])
G2.add_edges_from([(0,1), (0,2), (1,2)])
nx.set_node_attributes(G2, {0:'a', 1:'b', 2:'c'}, 'label')

G_nx = [G1, G2]

# Transforms list of NetworkX graphs into a list of GraKeL graphs
G = graph_from_networkx(G_nx, node_labels_tag='label')
print("2 - Node-labeled graphs transformed\n")


# Creates a list of two node-attributed graphs
G1 = nx.Graph()
G1.add_nodes_from([0,1,2])
G1.add_edges_from([(0,1), (1,2)])
nx.set_node_attributes(G1, {0:np.array([1.1, 0.8]),
    1:np.array([0.2, -0.3]), 2:np.array([0.9, 1.0])}, 'attributes')

G2 = nx.Graph()
G2.add_nodes_from([0,1,2])
G2.add_edges_from([(0,1), (0,2), (1,2)])
nx.set_node_attributes(G2, {0:np.array([1.8, 0.5]),
    1:np.array([-0.1, 0.2]), 2:np.array([2.3, 1.2])}, 'attributes')

G_nx = [G1, G2]

# Transforms list of NetworkX graphs into a list of GraKeL graphs
G = graph_from_networkx(G_nx, node_labels_tag='attributes')
print("3 - Node-attributed graphs transformed")

#######################

K = len(G2)
print(K)
ker = SubgraphMatching(ke=None, k=K)
G3 = nx.Graph()
G3.add_nodes_from([0,1,2])
G3.add_edges_from([(0,1), (0,2), (1,2)])
nx.set_node_attributes(G3, {0:1.8,
    1:0.2, 2:2.3}, 'attributes')

G_list = graph_from_networkx([G3], node_labels_tag='attributes')
G_list = [g for g in G_list]
ker.fit(G_list)
# res.initialize()

# print("fit")
# ker.fit([({(1, 2), (2, 3), (2, 1), (3, 2)},
#         {1: 'N', 2: 'C', 3: 'O'},
#         {(1, 2): ('N', 'C'), (2, 1): ('C', 'N'),
#          (2, 3): ('C', 'O'), (3, 2): ('O', 'C')})])

print("transform")

print(ker.transform(G_list))

print(ker.transform([({(1, 2), (2, 3), (3, 4), (3, 5), (5, 6),
                     (2, 1), (3, 2), (4, 3), (5, 3), (6, 5)},
                    {1: 'O', 2: 'C', 3: 'N', 4: 'C', 5: 'C', 6: 'O'},
                    {(1, 2): ('O', 'C'), (2, 3): ('C', 'N'),
                     (3, 4): ('N', 'C'), (3, 5): ('N', 'C'),
                     (5, 6): ('C', 'O'), (2, 1): ('C', 'O'),
                     (3, 2): ('N', 'C'), (4, 3): ('C', 'N'),
                     (5, 3): ('C', 'N'), (6, 5): ('O', 'C')})]))