import os
import pickle
import ast
import numpy as np
from adjustText import adjust_text
from matplotlib import pyplot as plt

from subgraph_matching_via_nn.data.paths import DATA_PATH
from subgraph_matching_via_nn.evaluation.mask_evaluation import CDFScoreService, \
    ConnectedInducedSubgraphRelativeHausdorffSimilarityCDFScore, \
    compute_relative_hausdorff_similarity_to_target_subgraph, induced_subgraph
from subgraph_matching_via_nn.evaluation.mask_metrics_constants import MaskMetricsConstants
from subgraph_matching_via_nn.full_graph_perturbation_analysis_experiment import load_sub_graph
from subgraph_matching_via_nn.graph_processors.graph_processors import GraphProcessor
from subgraph_matching_via_nn.utils.plot_services import PlotServices

BASELINE_CONFIGURATION_NAME = 'baseline'
BASE_CDF_CONFIGURATION_NAME = 'base_cdf'


def read_experiment_files(experiment_folder_path):
    dump_path = experiment_folder_path

    # read results file
    experiment_results_path = f"{dump_path}{os.sep}results.p"
    with open(experiment_results_path, 'rb') as f:
        experiment_results_map = pickle.load(f)

    # read config file
    experiment_config_path = f"{dump_path}{os.sep}config.p"
    with open(experiment_config_path, 'rb') as f:
        experiment_config_map = pickle.load(f)

    return experiment_results_map, experiment_config_map


def extract_dictionaries_from_file(file_path, header, base_perturbation_key):
    dictionaries = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(header):
                try:
                    # Find the start of the dictionary
                    dict_str = line.split(':', 1)[1].strip()
                    # Convert the string to a Python dictionary
                    dictionary = ast.literal_eval(dict_str)
                    dictionaries.append(dictionary)
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing dictionary: {e}")

    # add round nodes number as the key of the CDF

    return {base_perturbation_key + i: dictionary for i, dictionary in enumerate(dictionaries)}


def plot_score_graph(score_name, score_title_name):
    # Data
    configurations = sorted(list(experiment_folder_path_to_results_maps.keys()))
    config_name_to_id = {config_name: int(config_name) if config_name != BASELINE_CONFIGURATION_NAME else 0 for config_name in configurations}
    config_id_to_name = {int(config_name) if config_name != BASELINE_CONFIGURATION_NAME else 0 : config_name for config_name in configurations}
    # configurations_number = len(config_name_to_id)
    configurations_ids = sorted(list(config_name_to_id.values()))

    steps = sorted(list(experiment_results_map.keys()))
    x_values = steps

    config_id_to_y_values_map = {
        config_id: [experiment_folder_path_to_results_maps[config_id_to_name[config_id]][step_number][score_name] for step_number in steps]
        for config_id in configurations_ids
    }

    # Plotting
    colors = [
        'black',  # Color for baseline Configuration
        'blue',  # Color for Configuration 1
        'green',  # Color for Configuration 2
        'red',  # Color for Configuration 3
        'purple',  # Color for Configuration 4
        'orange',  # Color for Configuration 5
        'cyan',  # Color for Configuration 6
        'magenta',  # Color for Configuration 7
        'yellow',  # Color for Configuration 8
        'brown',  # Color for Configuration 9
        'pink',  # Color for Configuration 10
        'gray',  # Color for Configuration 11
        'olive',  # Color for Configuration 12
    ]
    plt.figure(figsize=(12, 8))
    for config_id, config_name in config_id_to_name.items():
        idx = config_id
        offset = idx * 1e-03

        try:
            config_map = experiment_folder_path_to_config_maps[config_name]
            print(f"plot ID #{idx} config name {config_name}. SPEC: {os.linesep} {config_map}")

            y = config_id_to_y_values_map[config_id]
            plt.plot(x_values, np.array(y) + offset, label=config_name, color=colors[idx])
        except:
            print(f"Error... skipping plot for configuration {config_id}")

    plt.xlabel('full graph size', fontsize=16)
    plt.ylabel(score_title_name, fontsize=16)
    plt.title(f'{score_title_name} perturbation analysis over different configurations', fontsize=16)
    plt.legend(title='config ID')
    plt.grid(True)
    plt.show()


def read_base_cdfs():
    nodes_overlap_cdf_extracted_maps_map = extract_dictionaries_from_file(
        f"{experiments_root_folder_path}{os.sep}{BASELINE_CONFIGURATION_NAME}{os.sep}log.txt",
        header='overlap CDF', base_perturbation_key=base_perturbation_key)
    rel_hausdorff_cdf_extracted_maps_map = extract_dictionaries_from_file(
        f"{experiments_root_folder_path}{os.sep}{BASELINE_CONFIGURATION_NAME}{os.sep}log.txt",
        header=f"{MaskMetricsConstants.RELATIVE_HAUSDORFF_SIMILARITY_CDF_SCORE_NAME} map",
        base_perturbation_key=base_perturbation_key)

    base_cdf_results_maps = {}
    for step_num, nodes_overlap_cdf_score_map in nodes_overlap_cdf_extracted_maps_map.items():
        base_cdf_results_maps[step_num] = {}
        base_cdf_results_maps[step_num][MaskMetricsConstants.OVERLAP_NODES_CDF_SCORE_NAME] = \
            nodes_overlap_cdf_score_map
        base_cdf_results_maps[step_num][MaskMetricsConstants.RELATIVE_HAUSDORFF_SIMILARITY_CDF_SCORE_NAME] = \
            rel_hausdorff_cdf_extracted_maps_map[step_num]

    return base_cdf_results_maps


def plot_cdf_base_comparison_graph(score_name, score_title_name, chosen_config_names, chosen_step_number):
    # chosen_step_number should make sense, i.e. don't take a perturbation step beyond the original size of the full graph
    # otherwise, the configurations are not comparable (different full graphs could be produced)

    # Data
    configurations = [BASELINE_CONFIGURATION_NAME] + chosen_config_names
    config_name_to_id = {config_name: int(config_name) if config_name != BASELINE_CONFIGURATION_NAME else 0 for
                         config_name in configurations}
    config_id_to_name = {int(config_name) if config_name != BASELINE_CONFIGURATION_NAME else 0: config_name for
                         config_name in configurations}

    # config_name_to_id = {config_name: i for i, config_name in enumerate(configurations)}
    # config_id_to_name = {i: config_name for i, config_name in enumerate(configurations)}
    configurations_ids = list(config_name_to_id.values())

    config_id_to_y_values_map = {
        config_id: [experiment_folder_path_to_results_maps[config_id_to_name[config_id]][step_number][score_name] for
                    step_number in [chosen_step_number]]
        for config_id in configurations_ids
    }

    base_cdf_chosen_step_map = base_cdf_results_maps[chosen_step_number][score_name]
    base_config_id = len(configurations_ids)
    config_id_to_name[base_config_id] = BASE_CDF_CONFIGURATION_NAME
    config_id_to_y_values_map[base_config_id] = base_cdf_chosen_step_map

    x_values = sorted(list(base_cdf_chosen_step_map.keys()))

    # Plotting
    colors = [
        'black',  # Color for baseline Configuration
        'blue',  # Color for Configuration 1
        'green',  # Color for Configuration 2
        'red',  # Color for Configuration 3
        'purple',  # Color for Configuration 4
        'orange',  # Color for Configuration 5
        'cyan',  # Color for Configuration 6
        'magenta',  # Color for Configuration 7
        'yellow',  # Color for Configuration 8
        'brown',  # Color for Configuration 9
        'pink',  # Color for Configuration 10
        'gray',  # Color for Configuration 11
        'olive',  # Color for Configuration 12
    ]
    base_cdf_color = 'crimson'
    plt.figure(figsize=(12, 8))
    texts = []
    for config_id, config_name in config_id_to_name.items():
        idx = config_id
        offset = idx * 1e-04

        try:
            y = config_id_to_y_values_map[config_id]

            if config_id == base_config_id:
                # plot all steps
                # extract config CDF values (currently a list of map)
                y = [y[x_value] for x_value in x_values]
                plt.plot(x_values, np.array(y) + offset, label=config_name, color=base_cdf_color)
            else:
                config_map = experiment_folder_path_to_config_maps[config_name]
                print(f"plot ID #{idx} config name {config_name}. SPEC: {os.linesep} {config_map}")

                # Marking the specific point
                precise_y_val = y[0]
                y_val = precise_y_val + offset
                x_val = None
                # get x_val according to CDF
                for cdf_x, cdf_val in base_cdf_chosen_step_map.items():
                    if cdf_val == precise_y_val:
                        x_val = cdf_x
                        break

                plt.scatter(x_val, y_val, color=colors[idx])
                texts.append(plt.text(x_val + offset, y_val + offset, config_name, color=colors[idx],
                         ha='right'))
        except:
            print(f"Error... skipping plot for configuration {config_id}")

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray'))

    plt.xlabel('base score', fontsize=16)
    plt.ylabel('CDF value', fontsize=16)
    plt.title(f'{score_title_name}', fontsize=16)
    # plt.legend(title='config ID')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    seed = 10  # for plotting
    plot_services = PlotServices(seed)

    experiments_root_folder_path = "C:\\Users\\kogan\\OneDrive\\Desktop\\nodes_number_analysis\\comp1_2\\subgraph0.p"
    base_perturbation_key = 8

    # experiments_root_folder_path = "C:\\Users\\kogan\\OneDrive\\Desktop\\nodes_number_analysis\\adder_8\\subgraph1.p"
    # base_perturbation_key = 16
    # chosen_config_names = ['7']
    chosen_step_number = 20

    experiment_folder_path_to_results_maps = {}
    experiment_folder_path_to_config_maps = {}
    for experiment_folder_path in os.listdir(experiments_root_folder_path):
        experiment_results_map, experiment_config_map = read_experiment_files(f"{experiments_root_folder_path}{os.sep}{experiment_folder_path}")
        experiment_folder_path_to_config_maps[experiment_folder_path] = experiment_config_map
        experiment_folder_path_to_results_maps[experiment_folder_path] = experiment_results_map

    # plot_score_graph(score_name='MaskMetricsConstants.F_BETA_SCORE_NAME', score_title_name="f-beta")

    plot_score_graph(score_name=MaskMetricsConstants.IS_CONNECTED_SCORE_NAME,
                     score_title_name="connectivity")
    plot_score_graph(score_name=MaskMetricsConstants.OVERLAP_NODES_SCORE_NAME,
                     score_title_name="correctly captured nodes #")
    plot_score_graph(score_name=MaskMetricsConstants.OVERLAP_NODES_CDF_SCORE_NAME,
                     score_title_name="correctly captured nodes # CDF")
    plot_score_graph(score_name=MaskMetricsConstants.RELATIVE_HAUSDORFF_SIMILARITY_SCORE_NAME,
                     score_title_name="relative Hausdorff similarity")
    plot_score_graph(score_name=MaskMetricsConstants.RELATIVE_HAUSDORFF_SIMILARITY_CDF_SCORE_NAME,
                     score_title_name="relative Hausdorff similarity CDF")

    # read all cdfs (one for each perturbation round)
    base_cdf_results_maps = read_base_cdfs()

    chosen_config_names = list(set(experiment_folder_path_to_config_maps.keys()).difference(set(BASELINE_CONFIGURATION_NAME)))

    plot_cdf_base_comparison_graph(score_name=MaskMetricsConstants.RELATIVE_HAUSDORFF_SIMILARITY_CDF_SCORE_NAME,
                     score_title_name="relative Hausdorff similarity CDF", chosen_config_names=chosen_config_names,
                                   chosen_step_number=chosen_step_number)

# # TODO: Fix hausdorff distances for both circuits [total 13 * 2 files]
    #
    # # read example
    # loader_params = {'data_path': DATA_PATH,
    #                  'g_full_path': f'comp1_2{os.sep}full_graph.p', #TODO:  adder_8
    #                  'g_sub_path': f'comp1_2{os.sep}subgraph0.p', #TODO subgraph1, adder_8
    #                  'is_use_features': False}
    # sub_graph = load_sub_graph(loader_params)
    # graph_processor = GraphProcessor(params={'to_line': False})  # , 'to_undirected': 'symmetrize'})
    # processed_sub_graph = graph_processor.pre_process(sub_graph)
    #
    # for step in sorted(list(experiment_results_map.keys())):
    #     if step > len(sub_graph.G):
    #         break
    #
    #     # calculate step hausdorff CDF map
    #     hausdorff_relative_similarity_cdf_scorer = ConnectedInducedSubgraphRelativeHausdorffSimilarityCDFScore(
    #         sub_graph)
    #     ordered_cdf_map = hausdorff_relative_similarity_cdf_scorer.calculate_cdf_histogram()
    #
    #     for experiment_folder_path, results_map in experiment_folder_path_to_results_maps.items():
    #         step_results_map = results_map[step]
    #
    #         # use subgraph + binarized solution to calculate rel hausdorff distance,
    #         binarized_solution = step_results_map[MaskMetricsConstants.BINARIZED_SOLUTION_RESULT_NAME]
    #         actual_subgraph = induced_subgraph(processed_sub_graph.G, binarized_solution)
    #
    #         hausdorff_relative_similarity, _ = compute_relative_hausdorff_similarity_to_target_subgraph(
    #             processed_sub_graph.G, processed_sub_graph.G_sub, actual_subgraph)
    #
    #         hausdorff_relative_similarity_cdf_score = CDFScoreService.get_cdf_score(ordered_cdf_map, hausdorff_relative_similarity,
    #                                                   is_complement=False)
    #
    #         # change hausdorff(2 scores)
    #         step_results_map[MaskMetricsConstants.RELATIVE_HAUSDORFF_SIMILARITY_SCORE_NAME] = hausdorff_relative_similarity
    #         step_results_map[MaskMetricsConstants.RELATIVE_HAUSDORFF_SIMILARITY_CDF_SCORE_NAME] = hausdorff_relative_similarity_cdf_score
    #
    # # after loop, override files
    # exit(0)