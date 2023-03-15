import torch
from sklearn.metrics import \
    accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, \
    jaccard_score
import numpy as np
import kmeans1d

from problem.base import E_from_v, indicator_from_v_np
from problem.spectral_subgraph_localization import SubgraphIsomorphismSolver


class MetricEvaluator:
    def __init__(self, A, subgraph_size=None):
        super().__init__()
        self.A = A
        self.subgraph_size = subgraph_size

    def accuracy(self, v_np, v_gt):
        self._validate_input(v_np)
        self._validate_input(v_gt)

        v_clustered, v_, s_clustered, s_ = \
            self._arrange_data(v_np=v_np, v_gt=v_gt)

        return accuracy_score(y_true=v_, y_pred=v_clustered), \
               accuracy_score(y_true=s_, y_pred=s_clustered)

    def balanced_accuracy(self, v_np, v_gt):
        self._validate_input(v_np)
        self._validate_input(v_gt)

        v_clustered, v_, e_clustered, e_ = \
            self._arrange_data(v_np=v_np, v_gt=v_gt)

        return balanced_accuracy_score(y_true=v_, y_pred=v_clustered), \
               balanced_accuracy_score(y_true=e_, y_pred=e_clustered)

    def confusion_matrix(self, v_np, v_gt):
        self._validate_input(v_np)
        self._validate_input(v_gt)

        v_clustered, v_, s_clustered, s_ = \
            self._arrange_data(v_np=v_np, v_gt=v_gt)

        return confusion_matrix(y_true=v_, y_pred=v_clustered, normalize='all'), \
               confusion_matrix(y_true=s_, y_pred=s_clustered, normalize='all')

    def jaccard_score(self, v_np, v_gt):
        self._validate_input(v_np)
        self._validate_input(v_gt)

        v_clustered, v_, s_clustered, s_ = \
            self._arrange_data(v_np=v_np, v_gt=v_gt)

        return jaccard_score(y_true=v_, y_pred=v_clustered, pos_label=1), \
               jaccard_score(y_true=s_, y_pred=s_clustered, pos_label=-1)

    def classification_report(self, v_np, v_gt):
        self._validate_input(v_np)
        self._validate_input(v_gt)

        v_clustered, v_, s_clustered, s_ = \
            self._arrange_data(v_np=v_np, v_gt=v_gt)
        v_target_names = ['v_in', 'v_out']
        E_target_names = ['E_', 'E_out']
        return classification_report(y_true=v_, y_pred=v_clustered,
                                     target_names=v_target_names), \
               classification_report(y_true=s_, y_pred=s_clustered,
                                     target_names=E_target_names)

    def _arrange_data(self, v_np, v_gt):

        # TODO: this part is alreadt implemented in SubgraphIsomorphismSolver.threshold.
        #  should make it abstract method and replace the following code
        # ------------------------------------------------------------
        v_clustered = np.zeros_like(v_np)
        ind = np.argsort(v_np)
        v_clustered[ind[self.subgraph_size:]] = 1
        # v_ = SubgraphIsomorphismSolver.indicator_from_v_np(v_np)
        # v_clustered, centroids = kmeans1d.cluster(v_, k=2)
        # ------------------------------------------------------------

        E_clustered, S_clustered = \
            E_from_v(torch.tensor(v_clustered), self.A)
        s_clustered = S_clustered[np.triu_indices(len(v_np))]

        v_ = indicator_from_v_np(v_gt.astype(int))
        E_gt, S_gt = E_from_v(torch.tensor(v_gt), self.A)
        S_ = S_gt.numpy()
        s_ = S_[np.triu_indices(len(v_np))]
        return np.ones_like(v_clustered) - v_clustered, \
               np.ones_like(v_clustered) - v_, \
               s_clustered, s_

    def evaluate(self, v_np, v_gt):
        self._validate_input(v_np)
        self._validate_input(v_gt)
        metrics = {"accuracy": self.accuracy(v_np, v_gt),
                   "balanced_accuracy": self.balanced_accuracy(v_np, v_gt),
                   "confusion_matrix": self.confusion_matrix(v_np, v_gt),
                   "classification_report": self.classification_report(v_np, v_gt),
                   "jaccard_score": self.jaccard_score(v_np, v_gt)}
        return metrics

    def _validate_input(self, v_np):
        if not (np.all(np.logical_or(v_np == 0, v_np == 1))):
            raise ValueError('v should contain only 0 and 1')

    @staticmethod
    def print(metrics, keys=None):
        if keys is None:
            keys = metrics[0].metric_keys()
        for key in keys:
            print(f"v {key} = {[metric[key][0] for metric in metrics]}")
            print(f"E {key} = {[metric[key][1] for metric in metrics]}")
