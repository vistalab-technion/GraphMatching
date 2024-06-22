import pickle
from prettytable import PrettyTable

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# Function to plot the graph with node labels and cluster annotations
import seaborn as sns
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_count_scatter(data, bins):

# Example data (replace with your data)
# data = [0.1, 0.2, 0.5, 1.2, 2.0, 3.5, 4.1, 5.2, 6.0, 7.5]

    # Define custom bins
    # bins = np.arange(0, 8, 1)  # Custom bins from 0 to 7 with width 1

    # Compute histogram with custom bins
    hist, bin_edges = np.histogram(data, bins=bins)

    # Calculate bin centers
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Plotting the scatter-like histogram with y-values in bin centers
    plt.figure(figsize=(8, 6))
    plt.scatter(bin_centers, hist, marker='o', color='blue', edgecolor='black')

    plt.xlabel('eigenvalue range')
    plt.ylabel('Count')
    plt.title('Eigenvalues histogram by range')

    # Customize y-axis ticks (histogram counts)
    plt.xticks(bins)
    plt.yticks(np.arange(0, max(hist)+1, 5))

    # Annotate a specific point (example: the first point)
    point_index = 0
    plt.annotate(f'zero eigenvalue multiplicity',
                 xy=(bin_centers[point_index], hist[point_index]),
                 xytext=(bin_centers[point_index] + 0.5, hist[point_index] + 1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    # # Circle the point
    # circle = plt.Circle((bin_centers[point_index], hist[point_index]), 0.3, color='red', fill=False)
    # plt.gca().add_artist(circle)

    plt.grid(True)
    plt.show()


def plot_graph_with_clusters(G, clusters, eigenvalue_indices, title):
    pos = nx.spring_layout(G, seed=42)
    cmap = plt.get_cmap('tab10', int(max(clusters) + 1))  # Use a categorical color map
    nx.draw(G, pos, with_labels=True, node_color=clusters, cmap=cmap, node_size=500, edge_color='gray', font_size=12)
    # for node, (x, y) in pos.items():
    #     plt.text(x, y + 0.05, str(eigenvalue_indices[node-1]), ha='center', va='center', fontsize=10)
    plt.title(title)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label='Cluster')
    plt.show()

# t = PrettyTable(['eigenvalue range', 'eigenvalue count'])
# t.add_row([0, 8])
# t.add_row(['(0, 1]', 0])
# t.add_row(['(1, 2]', 125])
# t.add_row(['(2, 3]', 123])
# t.add_row(['(3, 4]', 0])
# t.add_row(['(4, 5]', 0])
# print(t)
# exit(0)

# Create a sample graph (adjust as needed)
# G = nx.Graph()
# G.add_edges_from([(1, 2), (3, 4), (4, 5)])

G = pickle.load(open("RF_8BIT_GRAPH.P", 'rb'))

# # Plot the original graph
# plot_graph(G, "Original Graph")

# Calculate Laplacian matrix
L = nx.laplacian_matrix(G).toarray()

# Compute eigenvalues and eigenvectors of the Laplacian matrix
eigenvalues, eigenvectors = np.linalg.eigh(L)

# Sort eigenvalues in ascending order
eigenvalue_indices = np.argsort(eigenvalues)
eigenvalues = eigenvalues[eigenvalue_indices]
eigenvectors = eigenvectors[eigenvalue_indices]
print(eigenvalues)
# Interpretation 1: Graph Connectivity
EPS = 1e-10
num_connected_components = len(G) - np.count_nonzero(eigenvalues > EPS)  # Count non-zero eigenvalues (excluding numerical noise)
# print("Number of connected components (excluding numerical noise):", num_connected_components)


if num_connected_components == 1:
    print("Graph is fully connected.")
else:
    print("Graph is not fully connected. Number of connected components:", num_connected_components)

# Interpretation 2: Number of Connected Components


# Interpretation 3: Spectral Gap
spectral_gap = eigenvalues[1] - eigenvalues[0]
print("Spectral gap (lambda2 - lambda1):", spectral_gap)
print(eigenvalues)
# Determine clusters based on eigenvalues (simple thresholding for demonstration)
clusters = np.zeros(len(G.nodes))
num_clusters = 0
for i in range(len(eigenvalues)):
    if np.isclose(eigenvalues[i], 0):  # Check if eigenvalue is close to zero (considering numerical noise)
        clusters[eigenvectors[:, i] >= 0] = num_clusters
        num_clusters += 1
# print(num_clusters)
# print(clusters)

k = num_clusters  # Number of clusters (adjust as needed)
# spectral_features = eigenvectors[:, 1:k+1]  # Use the first k eigenvectors (excluding the zero eigenvalue)
#
# # Apply K-means clustering to spectral features
# kmeans = KMeans(n_clusters=k, random_state=42)
# clusters = kmeans.fit_predict(spectral_features)

# print(eigenvectors)
# second_smallest_eigenvalue_vectors = eigenvectors[:, eigenvalue_indices[1]]
# # print(second_smallest_eigenvalue_vectors)
# for i, value in enumerate(second_smallest_eigenvalue_vectors):
#     if value >= 0:
#         clusters[i] = 1
#     else:
#         clusters[i] = 0

# print(clusters)
# Plot the graph with cluster annotations
plot_graph_with_clusters(G, clusters, eigenvalue_indices, "Graph with Cluster Annotations")

list_size = len(G)
print(list_size)

plot_count_scatter(eigenvalues, bins=[-1e-6, EPS, 1, 2, 3, 4, 5])

# # Plotting the histogram
# plt.figure(figsize=(8, 6))
# # plt.hist(eigenvalues, bins=[-1e-6, EPS, 1, 2, 3, 4, 5], edgecolor='black', alpha=0.7, density=False)
# # sns.kdeplot(list(eigenvalues), fill=True, bw=0.05)
#
# # Compute KDE values directly
# kde_values = sns.kdeplot(eigenvalues, shade=True, bw=0.05).get_lines()[0].get_data()
#
# # Extract x and y values from KDE plot
# x_vals, y_vals = kde_values
#
# # Scale the y-values by the scale factor
# scaled_y_vals = y_vals * list_size
#
# # Plot the scaled KDE plot
# plt.fill_between(x_vals, scaled_y_vals, alpha=0.4, label=f'Scaled by {list_size}')
#
# plt.xlabel('Eigenvalue')
# plt.ylabel('Frequency')
# plt.title('Histogram of Eigenvalues')
# plt.grid(True)
#
# # Scale the y-axis by the list size for probability density
# # plt.gca().set_ylim([0, plt.gca().get_ylim()[1] * list_size])
#
# plt.show()