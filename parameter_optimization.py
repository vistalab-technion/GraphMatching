import optuna
import os
from datetime import datetime
import spectral_subgraph_localization as ssl
import pickle

def run_on_graphset():
    source_folder = 'data/benchmark/random_graphs/part_15_rest_20/'

    # print(source_folder)

    graph_dir = os.listdir(source_folder)

    now = datetime.now()


    # sh.copy(config_file, res_folder + '/' + config_file)

    list_full = []
    list_part = []
    list_nodes = []

    for file in graph_dir:
        # print(file)
        if 'full' in file:
            list_full.append(file)
        if 'part' in file:
            list_part.append(file)
        if 'nodes' in file:
            list_nodes.append(file)
    cg = 1
    # print(list_nodes)
    for graph in list_nodes:
        print(cg)
        cg = cg + 1
        cur_graph = graph[0:len(graph) - 10]
        part_nodes = source_folder + '/' + cur_graph + '_nodes.txt'
        part_graph = source_folder + '/' + cur_graph + '_part.txt'

        full_graphs = [g for g in list_full if cur_graph in g]

        for i in range(len(full_graphs)):
            graph_name = full_graphs[i][0:len(full_graphs[i]) - 9]
            full_graph = source_folder + '/' + full_graphs[i]
            disc_graph = source_folder + '/' + graph_name[0:-4] + '0000_full.txt'

            # print(full_graph)
            print(graph_name)

            if (os.stat(part_graph).st_size != 0):
                # try:

                func=lambda trial: ssl.opt_hyps(trial,full_graph,part_graph,part_nodes)
                study = optuna.create_study()
                study.optimize(func, n_trials=100)

                print()  # E.g. {'x': 2.002108042}
                print(study.best_value)

                res=study.best_params
                res["best_value"] = study.best_value
                with open('../results_tuning/'+graph_name+'.pickle', 'wb') as f:
                    pickle.dump(res, f)


                f = open('../results_tuning/'+graph_name+'.pickle', 'rb')
                cl = pickle.load(f)
                f.close()
                print(cl)


                # except:

        # [v_with gt,obj_w_gt,v_in_the_beginning, v_in_the_end, obj values,eigenvalues]=run_optimization(full_graph, part_graph,part_nodes, [big_list_of_params])


if __name__ == '__main__':

    run_on_graphset()