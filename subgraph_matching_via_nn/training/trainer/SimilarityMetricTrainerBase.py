import abc
import copy
import itertools as it
import math
import os
import time
from typing import Dict, Tuple, List, TypeVar
import torch_geometric as tg
import numpy as np
import torch
from livelossplot import PlotLosses
from matplotlib import pyplot as plt
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from subgraph_matching_via_nn.graph_metric_networks.graph_metric_nn import BaseGraphMetricNetwork
from subgraph_matching_via_nn.training.MarginLoss import MarginLoss
from subgraph_matching_via_nn.training.PairSampleBase import PairSampleBase
from subgraph_matching_via_nn.training.PairSampleInfo import Pair_Sample_Info
from subgraph_matching_via_nn.training.localization_grad_distance import LocalizationGradDistance
from subgraph_matching_via_nn.training.losses import get_pairs_batch_aggregated_distance
from subgraph_matching_via_nn.training.trainer.PairSampleInfo_with_S2VGraphs import PairSampleInfo_with_S2VGraphs


class SimilarityMetricTrainerBase(abc.ABC):

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

        self.stub_grad_distance = torch.tensor(float('nan'), requires_grad=False,
                                               device=self.device)
        self.inference_grad_distance = LocalizationGradDistance(problem_params,
                                                                solver_params)

    def _save_model(self, model, dump_path):
        model_state_dict = copy.deepcopy(model.state_dict())
        if dump_path is not None:
            torch.save(model_state_dict,
                       os.path.join(dump_path, 'best_model_state_dict.pt'))
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
        self.opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.get_params_list(), self.max_grad_norm)
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

    def _build_data_loaders(self, train_set, validation_set, collate_func):
        # TODO: this code is for supporting new samples, but it messes up overfitting a small dataset (loss is unstable)
        # data_sampler, train_set_with_sampler_labels = self.create_data_sampler(
        #     train_set, new_samples_amount)
        #
        # train_loader = tg.data.DataLoader(train_set_with_sampler_labels,
        #                                   batch_size=self.batch_size, sampler=data_sampler)

        train_loader = tg.data.DataLoader(train_set, batch_size=self.batch_size)
        train_loader.collate_fn = collate_func

        val_loader = tg.data.DataLoader(validation_set, batch_size=self.batch_size)
        val_loader.collate_fn = collate_func

        return train_loader, val_loader

    def train(self, train_samples_list, val_samples_list, new_samples_amount=0):
        """
        Train graph descriptor (model)
        Args:
            train_samples_list: list of graph pairs for training
            val_samples_list: list of graph pairs for validation
            new_samples_amount: how many of the train samples list were not trained previously. if 0 is passed, all train
            samples are treated as new samples. the new samples should be located at the end of the list

        Returns: all train losses list and all validation losses list

        """
        train_loader, val_loader = self.get_data_loaders(train_samples_list,
                                                         val_samples_list,
                                                         new_samples_amount)
        dump_base_path = self.dump_base_path

        dump_path = os.path.join(dump_base_path, str(time.time()))
        if not os.path.exists(dump_base_path):
            os.makedirs(dump_base_path)
        os.mkdir(dump_path)

        return self._train_loop(self.graph_similarity_module, train_loader, val_loader,
                                dump_path=dump_path)

    # Training loop, works on the graph pairs data loaders and the similarity model
    # if train_loss_convergence_threshold is None, rely on validation loss, cycle_patience, step_size_up and step_size_down
    # otherwise, rely on train loss, train_loss_convergence_threshold and successive_convergence_min_iterations_amount
    def _train_loop(self, model: BaseGraphMetricNetwork, train_loader, val_loader, dump_path=None):
        all_train_losses = []
        all_val_losses = []

        # define optimizer and scheduler
        if not self.init_optimizer(model):
            return all_train_losses, all_val_losses

        # init loop state
        liveloss = PlotLosses(mode='notebook')
        tqdm.write(f'dump path: {dump_path}')
        epoch_ctr = 0
        successive_convergence_epoch_ctr = 0
        best_val_loss = float('inf')
        train_loss_successive_convergence_counter = 0

        while (self.max_epochs is None) or (epoch_ctr < self.max_epochs):
            # init epoch state
            epoch_ctr += 1
            batch_index = 0

            epoch_train_loss = 0
            epoch_val_loss = 0

            for train_batch, val_batch in zip(train_loader, it.cycle(val_loader)):
                batch_index += 1

                # validation loss
                val_loss = self.calculate_loss_for_batch(val_batch,
                                                         is_train=False).item()
                epoch_val_loss += val_loss

                # train loss
                train_loss = self.calculate_loss_for_batch(train_batch, is_train=True)
                epoch_train_loss += train_loss.item()

                # optimization step
                self.optimization_step(model, train_loss)

            all_train_losses += [epoch_train_loss]
            all_val_losses += [epoch_val_loss]

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss

            # plot
            # self.plot_current_loss_history(all_train_losses, all_val_losses, start_time, epoch_ctr,
            #                           best_val_loss, dump_path)

            if epoch_ctr % self.solver_params['k_update_plot'] == 0:
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
                        model.load_state_dict(best_model_state_dict)
                        return all_train_losses, all_val_losses
                else:
                    # found better val loss
                    successive_convergence_epoch_ctr = 0
                    best_model_state_dict = self._save_model(model, dump_path)
            else:
                # check convergence in case relying on train loss
                if epoch_train_loss <= self.train_loss_convergence_threshold:
                    train_loss_successive_convergence_counter += 1
                    if train_loss_successive_convergence_counter >= self.successive_convergence_min_iterations_amount:
                        _ = self._save_model(model, dump_path)
                        return all_train_losses, all_val_losses
                else:
                    train_loss_successive_convergence_counter = 0

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

    T = TypeVar("T", Pair_Sample_Info, PairSampleInfo_with_S2VGraphs)

    @abc.abstractmethod
    def _get_pairs_list_loss(self, batch: Tuple[T, List[T]]) -> torch.Tensor:
        pass

    def calculate_loss_for_batch(self,
                                 batch: Tuple[T, List[T]],
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
                         val_set: List[Pair_Sample_Info], new_samples_amount) \
            -> (tg.data.DataLoader, tg.data.DataLoader):
        """
        create dataloaders for graph pairs based on graph wrapping data type
        Args:
            train_set:
            val_set:
            new_samples_amount:

        Returns:
            train DataLoader and validation DataLoader
        """
        pass