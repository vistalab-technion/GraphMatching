import ast
import numpy
import random
import matplotlib.pyplot as plt
import networkx as nx

from Common import PlotService
from Common.Math.AdjacencyMatrixUtils import from_matrix_to_graph
from Common.PlotService import color_graph_and_draw, plot_energy_curve
from APIImplementation.NetlistSmoothening.DependencyFrameGenerator import load_frame, generate_adjacency_matrices, \
    symmetrize_matrices
from Configuration.Config import Config
from APIImplementation.NetlistNoiseGeneration.GraphPerturbationService import remove_random_edge, \
    add_random_edge, get_mask_subcircuit_nodes, flip_edge
from APIImplementation.GraphDescriptorsKernel.GraphDescriptorsExtractor import measure_kernel_distance


def get_subcircuit_names(frame):
    gate_matrix_id_to_gate_name_map = get_gate_to_subcircuit_map(frame)
    subcircuit_names = set(gate_matrix_id_to_gate_name_map.values())

    return subcircuit_names

def generate_flipflop_subcircuit_masks(frame, hal_id_to_matrix_id):
    subcircuit_names = get_subcircuit_names(frame)
    gate_matrix_id_to_gate_name_map = get_gate_to_subcircuit_map(frame)

    ff_hal_gate_ids = frame.index.values
    masks = dict()
    for subcircuit_name in subcircuit_names:
        masks[subcircuit_name] = numpy.zeros((len(ff_hal_gate_ids), len(ff_hal_gate_ids)))

    for hal_gate_id in ff_hal_gate_ids:
        dep_ff_map = frame.loc[hal_gate_id]['dependencies']
        # it could have been been read from csv
        if type(dep_ff_map) is str:
            dep_ff_map = ast.literal_eval(dep_ff_map)
        source_gate_id_for_matrix = hal_id_to_matrix_id[hal_gate_id]

        for dep_gate_hal_id, dep_val in dep_ff_map.items():
            # if dep_ff_hal_gate_id not in hal_id_to_matrix_id:
            #	print('error for dep hal gate id {0}'.format(dep_ff_hal_gate_id))
            #	print(hal_id_to_matrix_id)
            #	return None
            dep_gate_id_for_matrix = hal_id_to_matrix_id[dep_gate_hal_id]

            if gate_matrix_id_to_gate_name_map[dep_gate_id_for_matrix] != gate_matrix_id_to_gate_name_map[source_gate_id_for_matrix]:
                continue
            subcircuit_name = gate_matrix_id_to_gate_name_map[dep_gate_id_for_matrix]
            subcircuit_mask = masks[subcircuit_name]
            subcircuit_mask[source_gate_id_for_matrix][dep_gate_id_for_matrix] = dep_val
            masks[subcircuit_name] = subcircuit_mask

    for mask in masks.values():
        if Config.ShouldSymmetrizeSmoothNetlist:
            symmetrize_matrices([mask])

    return masks


def get_gate_to_subcircuit_map(circuit_frame):
    gate_matrix_id_to_subcircuit_name_map = dict()
    ff_hal_gate_ids = circuit_frame.index.values

    hal_id_to_matrix_id = dict()
    matrix_id_to_hal_id = dict()
    matrix_id = 0
    for hal_gate_id in ff_hal_gate_ids:
        hal_id_to_matrix_id[hal_gate_id] = matrix_id
        matrix_id_to_hal_id[matrix_id] = hal_gate_id
        matrix_id += 1

    for hal_gate_id in ff_hal_gate_ids:
        matrix_id = hal_id_to_matrix_id[hal_gate_id]
        gate_name = circuit_frame.loc[hal_gate_id]['gate_name']
        gate_matrix_id_to_subcircuit_name_map[matrix_id] = gate_name[:gate_name.index('_')]

    return gate_matrix_id_to_subcircuit_name_map


def get_all_circuit_subcircuit_masks(frame, subcircuit_name):
    ff_adjacency_matrices, _, gate_hal_id_to_matrix_id_maps = generate_adjacency_matrices([frame], [None],
                                                           scan_noise_threshold=None)
    circuit_matrix = ff_adjacency_matrices[0]

    if subcircuit_name == "" or subcircuit_name is None:
        return False, (circuit_matrix, circuit_matrix)

    return True, (circuit_matrix, generate_flipflop_subcircuit_masks(frame, gate_hal_id_to_matrix_id_maps[0]))


def extract_subcircuit_out_of_circuit(frame, subcircuit_name):
    flag, masks = get_all_circuit_subcircuit_masks(frame, subcircuit_name)
    if not flag:
        return masks

    circuit_matrix, masks = masks

    print(len(masks), " masks were extracted")
    subcircuit_mask = masks[subcircuit_name] #next(iter(masks.values()))

    return circuit_matrix, subcircuit_mask


def simulate_perturbed_mask_energy_plain(frame_path):
    frame = load_frame(frame_path)
    ff_adjacency_matrices, _, _ = generate_adjacency_matrices([frame], [None],
                                                                      scan_noise_threshold=None)
    ff_adjacency_matrix = ff_adjacency_matrices[0]
    ground_truth_matrix = numpy.copy(ff_adjacency_matrix)
    ground_truth_graph = from_matrix_to_graph(ground_truth_matrix)

    energy = measure_kernel_distance(ground_truth_graph, ground_truth_graph, kernel="GK-WL")
    nx.draw(ground_truth_graph)
    plt.show()

    perturbed_matrix = ff_adjacency_matrix
    x = [0]
    y = [energy]
    for i in range(15):
        #perturbed_matrix = change_adjacency_matrix_cell(perturbed_matrix)
        rand_action = random.randint(0,1)
        if rand_action == 0:
            if not remove_random_edge(perturbed_matrix):
                break
        elif rand_action == 1:
            if not add_random_edge(perturbed_matrix):
                break
        x.append(i+1)
        perturbed_graph = from_matrix_to_graph(perturbed_matrix)
        energy = measure_kernel_distance(ground_truth_graph, perturbed_graph, kernel="GK-WL")
        y.append(energy)

    nx.draw(perturbed_graph)
    plt.show()

    fig, ax = plt.subplots(1)

    # plot the data
    ax.plot(x, y)
    plt.show()


def get_subcircuit_boundary_edges(circuit_matrix, subcircuit_mask, is_potential_add_edges=1):
    nodes_amount = len(circuit_matrix)

    subcircuit_mask_node_indices = get_mask_subcircuit_nodes(subcircuit_mask)
    circuit_node_indices = get_mask_subcircuit_nodes(circuit_matrix)

    boundary_edges = [(i, j) for i in range(nodes_amount) for j in range(nodes_amount) if
                      ((circuit_matrix[i][j] != 0)
                      and (
                      (
                              (i in circuit_node_indices) and
                              (i not in subcircuit_mask_node_indices) and
                              (j in subcircuit_mask_node_indices)

                      )
                                  or
                      (
                          (not Config.ShouldSymmetrizeSmoothNetlist)
                            and
                          (
                                  (j in circuit_node_indices) and
                                  (j not in subcircuit_mask_node_indices) and
                                  (i in subcircuit_mask_node_indices)
                          )
                      )
                            ))]

    return boundary_edges


def perturb_mask_around_subcircuit(frame_path, subcircuit_name, amount, max_perturb_actions=1, is_uniform_distribtion=True, kernel=None):
    perturbed_matrices = []
    survived_nodes = []

    for i in range(amount):
        perturb_amount = max_perturb_actions
        if is_uniform_distribtion:
            perturb_amount = random.randint(1, max_perturb_actions)

        perturbed_matrix, _, _, _, _, _ = perturb_clutter(frame_path, subcircuit_name, is_track_energy=False, iter_num=perturb_amount, kernel=kernel)

        survived_nodes.append(get_mask_subcircuit_nodes(perturbed_matrix))
        perturbed_matrices.append(perturbed_matrix)

    return perturbed_matrices, survived_nodes


def perturb_clutter(frame, subcircuit_name, is_track_energy, iter_num=12, kernel=None, is_plot=True):
    #for energy tracking
    circuit_matrix, subcircuit_mask = extract_subcircuit_out_of_circuit(frame, subcircuit_name)
    perturbed_matrix = subcircuit_mask

    ground_truth_matrix = numpy.copy(subcircuit_mask)
    ground_truth_graph = from_matrix_to_graph(ground_truth_matrix)
    perturbed_graph = ground_truth_graph
    energy = measure_kernel_distance(ground_truth_graph, ground_truth_graph, kernel=kernel)

    perturbed_graphs = []
    graph_images_to_plot = []
    graph_positions_to_plot = []
    x = [0]
    y = [energy]
    ######################

    boundary_edges = get_subcircuit_boundary_edges(circuit_matrix, perturbed_matrix)
    for i in range(iter_num):
        if len(boundary_edges) == 0:
            break

        if is_track_energy:
            perturbed_graphs.append(perturbed_graph)
            x_pos = i
            if is_plot:
                if i % int(iter_num / 4) == 0:
                    color_graph_and_draw(perturbed_graph, ground_truth_matrix)
                    graph_file_name = "Graph_clutter_tmp %d .png" % i
                    graph_images_to_plot.append(graph_file_name)
                    graph_positions_to_plot.append((x_pos, energy - PlotService.y_plot_offset))
                    plt.savefig(graph_file_name, format="PNG")

        boundary_edge = boundary_edges[random.randint(0, len(boundary_edges) - 1)]
        boundary_edges.remove(boundary_edge)

        flip_edge(perturbed_matrix, boundary_edge[0], boundary_edge[1])
        if Config.ShouldSymmetrizeSmoothNetlist:
            flip_edge(perturbed_matrix, boundary_edge[1], boundary_edge[0])

        if is_track_energy:
            perturbed_graph = from_matrix_to_graph(perturbed_matrix)
            energy = measure_kernel_distance(ground_truth_graph, perturbed_graph, kernel=kernel)
            x.append(x_pos+1)
            y.append(energy)

    if is_track_energy and is_plot:
        # labels = [str(node_index) for node_index in range(len(perturbed_matrix))]
        # graph_frame = pandas.DataFrame(perturbed_matrix, index=labels, columns=labels)
        # nx.draw_networkx(nx.from_pandas_adjacency(graph_frame))
        # nx.draw(perturbed_graph)
        # plt.show()

        color_graph_and_draw(perturbed_graph, ground_truth_matrix)
        #plt.show()
        plt.close()

    return perturbed_matrix, perturbed_graphs, graph_images_to_plot, graph_positions_to_plot, x, y


def simulate_perturbed_mask_energy_around_subcircuit_without_plot(frame_path, subcircuit_name, iter_num, kernel, is_plot=True):
    frame = load_frame(frame_path)
    _, perturbed_graphs, graph_images_to_plot, graph_positions_to_plot, x, y = perturb_clutter(frame, subcircuit_name,
                                                                                               is_track_energy=True,
                                                                                               iter_num=iter_num,
                                                                                               kernel=kernel,
                                                                                               is_plot=is_plot)
    return perturbed_graphs, graph_images_to_plot, graph_positions_to_plot, x, y


def simulate_perturbed_mask_energy_around_subcircuit(netlist_name, frame_path, subcircuit_name, iter_num, kernel, is_for_show=False):
    perturbed_graphs, graph_images_to_plot, graph_positions_to_plot, x, y = simulate_perturbed_mask_energy_around_subcircuit_without_plot(frame_path, subcircuit_name, iter_num, kernel)

    plot_energy_curve(netlist_name, subcircuit_name, graph_images_to_plot, graph_positions_to_plot, x, y, "cluttered", is_for_show=is_for_show)

    return perturbed_graphs