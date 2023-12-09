import abc
import copy
import math
import os
import pickle
import sys
import time
from typing import Dict, Tuple, List, TypeVar
import torch_geometric as tg
import numpy as np
import torch
from livelossplot import PlotLosses
from matplotlib import pyplot as plt
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from powerful_gnns.models.graphcnn import GraphCNN
from subgraph_matching_via_nn.graph_metric_networks.graph_metric_nn import BaseGraphMetricNetwork
from subgraph_matching_via_nn.training.MarginLoss import MarginLoss
from subgraph_matching_via_nn.training.PairSampleBase import PairSampleBase
from subgraph_matching_via_nn.training.PairSampleInfo import Pair_Sample_Info
from subgraph_matching_via_nn.training.localization_grad_distance import LocalizationGradDistance
from subgraph_matching_via_nn.training.losses import get_pairs_batch_aggregated_distance
from subgraph_matching_via_nn.training.trainer.PairSampleInfo_with_S2VGraphs import PairSampleInfo_with_S2VGraphs
from subgraph_matching_via_nn.training.trainer.dataset_partitioning import average_gradients, partition_dataset, \
    split_dataset

import torch.distributed as dist
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')


class SimilarityMetricTrainerBase(abc.ABC):
    SENTINEL = 'SENTINEL'

    @staticmethod
    def init_process(rank, size, fn, q, device_id, train_loader, val_loader, backend='gloo'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=rank, world_size=size)

        fn(device_id, q, train_loader, val_loader)

    def reset_modules_parameters(self, device_id):
        torch.manual_seed(1234)

        # reset tensor values before creating new process,
        # as the combination with CUDA usage sets them to zero for some reason,
        # see https://discuss.pytorch.org/t/multiprocessing-cause-models-parameters-all-become-to-0-0/148183
        self.stub_grad_distance = torch.tensor(float('nan'), requires_grad=False,
                                               device=device_id)

        gnn_model = self.graph_similarity_module.embedding_networks[0].gnn_model
        gnn_model.load_state_dict(torch.load(
            self.original_model_path,
            map_location=torch.device(device_id)))
        self.graph_similarity_module.to(device_id).requires_grad_(True)
        gnn_model.move_buffers_to_device()

    def _training_worker_run_func(self, device_id, q, train_loader_path, val_loader_path):
        self.device = device_id
        self.reset_modules_parameters(device_id)

        with open(train_loader_path, 'rb') as file:
            train_loader = pickle.load(file)

        if val_loader_path is None:
            val_loader = None
        else:
            with open(val_loader_path, 'rb') as file:
                val_loader = pickle.load(file)

        if self.device == "cpu":
            train_loader.pin_memory = True
            if val_loader is not None:
                val_loader.pin_memory = True

        # for batch in train_loader:
        #     for pair in batch:
        #         for s2v_graph in pair.s2v_graphs:
        #             s2v_graph.to(device=self.device)
        # if val_loader is not None:
        #     for batch in val_loader:
        #         for pair in batch:
        #             for s2v_graph in pair.s2v_graphs:
        #                 s2v_graph.to(device=self.device)

        return self._train_loop(self.graph_similarity_module, train_loader, val_loader, q)

    def __init__(self, graph_similarity_module: BaseGraphMetricNetwork, dump_base_path: str,
                 problem_params: Dict, solver_params: Dict):
        # super().__init__()
        self.graph_similarity_module = graph_similarity_module
        self.dump_base_path = dump_base_path
        self.problem_params = problem_params
        self.solver_params = solver_params

        self.device = self.solver_params["device"]
        self.lr = self.solver_params["lr"]
        self.weight_decay = self.solver_params["weight_decay"]
        self.cycle_patience = self.solver_params["cycle_patience"]
        self.step_size_up = self.solver_params["step_size_up"]
        self.step_size_down = self.solver_params["step_size_down"]
        self.max_grad_norm = self.solver_params["max_grad_norm"]

        self.train_loss_convergence_threshold = self.solver_params[
            "train_loss_convergence_threshold"]
        self.successive_convergence_min_iterations_amount = self.solver_params[
            "successive_convergence_min_iterations_amount"]

        # batching is not supported for nx.graph-s in general, so the trained model must support batching! otherwise, set batch_size to 1
        self.batch_size = self.solver_params["batch_size"] #1

        self.opt = None
        self.lrs = None
        self.max_epochs = self.solver_params['max_epochs']

        self.graph_similarity_loss_function = MarginLoss(
            solver_params['margin_loss_margin_value'])

        self.stub_grad_distance = None
        self.inference_grad_distance = LocalizationGradDistance(problem_params,
                                                                solver_params)
                                                                
        self.previous_train_loader_paths = None
        self.previous_val_loader_paths = None

        dump_base_path = f".{os.sep}mp"
        dump_path = os.path.join(dump_base_path, str(time.time()))
        if not os.path.exists(dump_base_path):
            os.makedirs(dump_base_path)
        os.mkdir(dump_path)
        original_model_file_name = "original_model.pt"

        model = GraphCNN(num_layers=5, num_mlp_layers=2, input_dim=1, hidden_dim=64, output_dim=2,
                         final_dropout=0.5, learn_eps=False, graph_pooling_type="sum", neighbor_pooling_type="sum",
                         device='cpu')

        self._save_model(model, dump_path, original_model_file_name)

        self.original_model_path = os.path.join(dump_path, original_model_file_name)


    def _save_model(self, model, dump_path, file_name='best_model_state_dict.pt'):
        model_state_dict = copy.deepcopy(model.state_dict())
        if dump_path is not None:
            torch.save(model_state_dict,
                       os.path.join(dump_path, file_name))
        return model_state_dict

    def _save_model_stats(self, model, dump_path, all_train_losses, all_val_losses):
        if dump_path is not None:
            torch.save(all_train_losses, os.path.join(dump_path, 'train_losses.pt'))
            torch.save(all_val_losses, os.path.join(dump_path, 'val_losses.pt'))
            #torch.save(self.lrs, os.path.join(dump_path, 'lrs.pt'))
            torch.save(model.state_dict(), os.path.join(dump_path, 'model.pt'))

            train_losses = all_train_losses[-1]
            val_losses = all_val_losses[-1]
            with open(os.path.join(dump_path, 'losses.txt'), 'a') as file:
                file.write(
                    f'train:{np.average(train_losses)}\t val:{np.average(val_losses)}\n')

    def optimization_step(self, model, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.get_params_list(), self.max_grad_norm)
        average_gradients(model)

        self.opt.step()
        self.lrs.step()

    def init_optimizer(self, model: BaseGraphMetricNetwork):
        model_parameters = model.get_params_list()
        if len(model_parameters) == 0:
            return False

        # self.opt = torch.optim.AdamW(model_parameters, lr=self.lr,
        #                              weight_decay=self.weight_decay)


        self.opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        # self.lrs = torch.optim.lr_scheduler.StepLR(self.opt, step_size=50, gamma=0.5)
        # self.lrs = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100)
        self.lrs = torch.optim.lr_scheduler.CyclicLR(self.opt, base_lr=0,
                                                     max_lr=self.lr,
                                                     step_size_up=self.step_size_up,
                                                     step_size_down=self.step_size_down,
                                                     cycle_momentum=False)

        return True

    def __create_data_loaders_lists(self, full_data_set, collate_func, device_ids):
        n_partitions = len(device_ids)
        num_workers = 0  # self.solver_params['num_workers']
        data_loaders = []

        datasets, bsz = split_dataset(full_data_set, self.batch_size, n_partitions)
        for dataset in datasets:
            loader = tg.loader.DataLoader(dataset, batch_size=bsz, shuffle=True, num_workers=num_workers)
            loader.collate_fn = collate_func
            data_loaders.append(loader)

        return data_loaders

    def _build_data_loaders(self, train_set, validation_set, collate_func, device_ids):
        # TODO: this code is for supporting new samples, but it messes up overfitting a small dataset (loss is unstable)
        # data_sampler, train_set_with_sampler_labels = self.create_data_sampler(
        #     train_set, new_samples_amount)
        #
        # train_loader = tg.data.DataLoader(train_set_with_sampler_labels,
        #                                   batch_size=self.batch_size, sampler=data_sampler)

        n_partitions = len(device_ids)

        train_data_loaders = self.__create_data_loaders_lists(train_set, collate_func, device_ids)
        if len(validation_set) != 0:
            val_data_loaders = self.__create_data_loaders_lists(validation_set, collate_func, device_ids)
        else:
            val_data_loaders = [None for rank in range(n_partitions)]

        # partition, bsz = partition_dataset(train_set, self.batch_size)
        
        return train_data_loaders, val_data_loaders

    def __save_dataloaders_locally(self, dataloader_list):
        dl_paths = []

        dump_base_path = f".{os.sep}dataloaders"
        if not os.path.exists(dump_base_path):
            os.makedirs(dump_base_path)

        for dataloader in dataloader_list:
            dl_path = os.path.join(dump_base_path, f"{str(time.time())}.p")
            with open(dl_path, 'wb') as f:
                pickle.dump(dataloader, f)
            dl_paths.append(dl_path)
        return dl_paths

    def set_existing_data_loader_paths(self, train_dataloader_paths, val_dataloader_paths):
        self.previous_train_loader_paths = train_dataloader_paths
        self.previous_val_loader_paths = val_dataloader_paths

    def train(self, processes_device_ids, use_existing_data_loaders=False, train_samples_list=[], val_samples_list=[], new_samples_amount=0):
        """
        Train graph descriptor (model)
        Args:
            use_existing_data_loaders: boolean flag to indicate if to build data loaders or use the existing ones
            train_samples_list: list of graph pairs for training
            val_samples_list: list of graph pairs for validation
            new_samples_amount: how many of the train samples list were not trained previously. if 0 is passed, all train
            samples are treated as new samples. the new samples should be located at the end of the list

        Returns: all train losses list and all validation losses list

        """
        self.graph_similarity_module.train()

        size = len(processes_device_ids)
        processes = []
        queues = []

        if not use_existing_data_loaders:
            train_loaders, val_loaders = self.get_data_loaders(train_samples_list, val_samples_list,
                                                             new_samples_amount, processes_device_ids)
            self.previous_train_loader_paths = self.__save_dataloaders_locally(train_loaders)
            self.previous_val_loader_paths = self.__save_dataloaders_locally(val_loaders)

            # clear memory
            train_loaders = val_loaders = None
            train_samples_list = val_samples_list = None
            torch.cuda.empty_cache()
            print("Data loaders were built and saved locally")
        else:
            print("Using existing data loaders")

        for rank, device_id in enumerate(processes_device_ids):
            train_loader_path = self.previous_train_loader_paths[rank]
            val_loader_path = self.previous_val_loader_paths[rank]
            q = mp.Queue()
            p = mp.Process(target=SimilarityMetricTrainerBase.init_process,
                           args=(rank, size, self._training_worker_run_func, q, device_id,
                                 train_loader_path, val_loader_path))
            p.start()

            queues.append(q)
            processes.append(p)

        dump_base_path = self.dump_base_path
        dump_path = os.path.join(dump_base_path, f"{str(time.time())}")
        if not os.path.exists(dump_base_path):
            os.makedirs(dump_base_path)
        os.mkdir(dump_path)

        self.monitoring_training(queues, dump_path)

        for i, p in enumerate(processes):
            p.join()
            print(f"training worker process #{i} finished")
        print("finished monitoring training")

    def monitoring_training(self, queues, dump_path=None):
        all_train_losses = []
        all_val_losses = []
        successive_convergence_epoch_ctr = 0
        train_loss_successive_convergence_counter = 0
        epoch_ctr = 0
        best_val_loss = float('inf')

        liveloss = PlotLosses(mode='notebook')
        n_queues = len(queues)

        while True:
            epoch_train_loss = 0
            epoch_val_loss = 0
            last_monitoring_update_list = None
            monitoring_update_lists = []

            for q in queues:
                last_monitoring_update_list = q.get()

                if last_monitoring_update_list == SimilarityMetricTrainerBase.SENTINEL:
                    print(f"Monitoring process is exiting, epochs={epoch_ctr}")
                    break
                monitoring_update_lists.append(last_monitoring_update_list)

            if last_monitoring_update_list == SimilarityMetricTrainerBase.SENTINEL:
                break

            update_list_size = len(monitoring_update_lists[0])
            for update_list_index in range(update_list_size):
                for q_index in range(n_queues):

                    worker_train_loss, worker_val_loss = monitoring_update_lists[q_index][update_list_index]
                    epoch_train_loss += worker_train_loss
                    epoch_val_loss += worker_val_loss

                    epoch_ctr += 1
                    all_train_losses += [epoch_train_loss]
                    all_val_losses += [epoch_val_loss]

                    if epoch_val_loss < best_val_loss:
                        best_val_loss = epoch_val_loss

                    if epoch_ctr % self.solver_params['k_update_plot'] == 0:
                        # self.plot_current_loss_history(all_train_losses, all_val_losses, start_time, epoch_ctr,
                        #                           best_val_loss, dump_path)

                        print(f"finished epoch {epoch_ctr} best_val_loss = {best_val_loss}")
                        liveloss.update({'epoch train loss': epoch_train_loss})
                        liveloss.send()

                        # save updated best model (with stats) post epoch
                        # self._save_model_stats(model, dump_path, all_train_losses, all_val_losses)

                    if self.train_loss_convergence_threshold is None:
                        # check convergence in case relying on validation loss
                        if epoch_val_loss >= best_val_loss:
                            successive_convergence_epoch_ctr += 1
                            if successive_convergence_epoch_ctr > self.cycle_patience * (
                                    self.step_size_up + self.step_size_down):
                                # model.load_state_dict(best_model_state_dict) #TODO
                                return all_train_losses, all_val_losses
                        else:
                            # found better val loss
                            successive_convergence_epoch_ctr = 0
                            # best_model_state_dict = self._save_model(model, dump_path) #TODO
                    else:
                        # check convergence in case relying on train loss
                        if epoch_train_loss <= self.train_loss_convergence_threshold:
                            train_loss_successive_convergence_counter += 1
                            if train_loss_successive_convergence_counter >= self.successive_convergence_min_iterations_amount:
                                # _ = self._save_model(model, dump_path) #TODO - model should be tracked, but avoid sending it between process (too heavy)
                                return all_train_losses, all_val_losses
                        else:
                            train_loss_successive_convergence_counter = 0

        return all_train_losses, all_val_losses

    def __terminate_worker_process(self, q):
        rank = dist.get_rank()
        print(f"worker process {rank} is terminating at {time.strftime('%H:%M:%S', time.localtime())}")
        sys.stdout.flush()
        q.put(SimilarityMetricTrainerBase.SENTINEL)

    def __move_batch_to_device(self, graphs_batch):
        new_graphs_batch = [PairSampleInfo_with_S2VGraphs(
            pair.pair_sample_info,
            (
                pair.s2v_graphs[0].to(device=self.device, non_blocking=True),
                pair.s2v_graphs[1].to(device=self.device, non_blocking=True)
            )
        )
            for pair in graphs_batch]
        return new_graphs_batch

    # Training loop, works on the graph pairs data loaders and the similarity model
    # if train_loss_convergence_threshold is None, rely on validation loss, cycle_patience, step_size_up and step_size_down
    # otherwise, rely on train loss, train_loss_convergence_threshold and successive_convergence_min_iterations_amount
    def _train_loop(self, model: BaseGraphMetricNetwork, train_loader, val_loader, q):
        rank = dist.get_rank()
        print(f"worker process #{rank} is starting at {time.strftime('%H:%M:%S', time.localtime())}", flush=True)

        monitoring_update_epochs_pace = self.solver_params['train_monitoring_epochs_pace']

        # define optimizer and scheduler
        if not self.init_optimizer(model):
            self.__terminate_worker_process(q)
            return

        # init loop state
        epoch_ctr = 0
        # actual_batch_size = train_loader.batch_size
        # num_batches = math.ceil(len(train_loader.dataset) / float(actual_batch_size))

        monitoring_update_list = []

        while (self.max_epochs is None) or (epoch_ctr < self.max_epochs):
            # init epoch state
            epoch_ctr += 1
            batch_index = 0

            epoch_train_loss = 0
            epoch_val_loss = 0

            # print('Rank ', rank, ', epoch ',
            #       epoch_ctr)
            for train_batch in train_loader:
                #print('Rank ', rank, f", Batch #{batch_index}", flush=True)
                batch_index += 1

                train_batch = self.__move_batch_to_device(train_batch)

                # train loss
                train_loss = self.calculate_loss_for_batch(train_batch, is_train=True)
                epoch_train_loss += train_loss.item()

                # optimization step
                self.optimization_step(model, train_loss)

            print('Rank ', rank, ', epoch ', epoch_ctr)
            #      , ': ', epoch_train_loss / num_batches)
            sys.stdout.flush()

            if val_loader is None:
                epoch_val_loss = epoch_train_loss
            else:
                for val_batch in val_loader:
                    val_batch = self.__move_batch_to_device(val_batch)
                    # validation loss
                    val_loss = self.calculate_loss_for_batch(val_batch,
                                                             is_train=False)
                    epoch_val_loss += val_loss.item()

            # avoid deadlock when there are multiple processes -> make sure all processes finish together epoch wise
            monitoring_update_list.append((epoch_train_loss, epoch_val_loss))
            if epoch_ctr % monitoring_update_epochs_pace == 0:
                q.put(monitoring_update_list)
                monitoring_update_list = []

        if len(monitoring_update_list) != 0:
            q.put(monitoring_update_list)

        print("rank", rank, ", epochs trained: ", epoch_ctr)
        self.__terminate_worker_process(q)

    def create_data_sampler(self, train_set, new_examples_set_size):
        n = len(train_set)
        x = new_examples_set_size
        b = n - x

        p_new_label = (math.pow(x, 2) + 2 * b * x) / (
                2 * math.pow(x, 2) + 3 * b * x + math.pow(b, 2))
        w_new_label = p_new_label / (1 - p_new_label)

        weights = [1 if i < b else w_new_label for i, item in enumerate(train_set)]
        train_set_with_sampler_labels = [(item, 0) if i < b else (item, 1) for i, item
                                         in enumerate(train_set)]
        return WeightedRandomSampler(weights=weights, num_samples=n,
                                     replacement=True), train_set_with_sampler_labels

    def get_grad_distance(self, pair_sample_info: PairSampleBase,
                          localization_object: object):
        if localization_object is not None:
            #TODO: Avoid memory sync between cpu to CUDA, if possible [remove .to(self.device)]
            pair_sample_info.localization_object = localization_object.to(self.device)

        # this part takes relatively a long time
        # (we can probably optimize it way if we save it when the example is first generated)
        grad_distance = self.inference_grad_distance.compute_grad_distance(
            pair_sample_info)
        return grad_distance

    def get_aggregated_pairs_batch_distance(self, emb_distances, batch: Tuple[
        Pair_Sample_Info, List[Pair_Sample_Info]]):
        _, samples = batch
        grad_distances = [self.stub_grad_distance if pair.localization_object is None
                          else self.get_grad_distance(pair, pair.localization_object)
                          for pair in samples]

        is_negative_sample_flags = [pair.is_negative_sample for pair in samples]
        return get_pairs_batch_aggregated_distance(self.graph_similarity_loss_function,
                                                   emb_distances, grad_distances,
                                                   is_negative_sample_flags)

    def plot_current_loss_history(self, all_train_losses, all_val_losses, start_time,
                                  epoch_ctr, epoch_best_val_loss,
                                  dump_path):

        plt.figure()
        plt.plot(all_train_losses, label='train')
        plt.plot(all_val_losses, label='val')
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('loss')

        val_losses = all_val_losses[-1]

        if val_losses[0] <= 1:
            plt.ylim((0, 1))
        elif val_losses[0] <= 5:
            plt.ylim((0, 5))
        elif val_losses[0] <= 10:
            plt.ylim((0, 10))
        elif val_losses[0] <= 100:
            plt.ylim((0, 100))

        val_loss = val_losses[-1]
        plt.title(
            f'epoch:{epoch_ctr} | time:{int(time.time() - start_time)}s | '
            f'val:{val_loss:.3} | best:{epoch_best_val_loss:.3} | '
            f'patience:{self.bctr // (self.step_size_up + self.step_size_down)}/{self.cycle_patience}')
        # if utils.is_notebook() and not dump_path:
        #     ipy.display.clear_output(wait=True)
        #     plt.show()
        # else:
        assert dump_path is not None
        plt.savefig(os.path.join(dump_path, 'loss_curve.png'))
        plt.close()

    T = TypeVar("T")

    @abc.abstractmethod
    def _get_pairs_list_loss(self, batch: T) -> torch.Tensor:
        pass

    def calculate_loss_for_batch(self,
                                 batch: T,
                                 is_train: bool) \
            -> torch.Tensor:
        """
        Calculate entire batch graphs pairs losses
        Args:
            batch: a tuple of a single pair holding the entire batch data (Pair_Sample_Info) and a list of the
             pairs seperated (List[Pair_Sample_Info]
            is_train: True for training, False for validation

        Returns:
            Total batch loss
        """
        batch_loss = None

        if is_train:
            batch_loss = self._get_pairs_list_loss(batch)
        else:
            with torch.no_grad():
                batch_loss = self._get_pairs_list_loss(batch)

        return batch_loss

    @abc.abstractmethod
    def get_data_loaders(self, train_set: List[Pair_Sample_Info],
                         val_set: List[Pair_Sample_Info], new_samples_amount: int, device_ids: List[int]) \
            -> (tg.data.DataLoader, tg.data.DataLoader):
        """
        create dataloaders for graph pairs based on graph wrapping data type
        Args:
            train_set:
            val_set:
            new_samples_amount:
            device_ids:

        Returns:
            train DataLoader and validation DataLoader
        """
        pass