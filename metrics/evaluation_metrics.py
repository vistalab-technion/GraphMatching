import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score,jaccard_score
import numpy as np
import kmeans1d
from problem.spectral_subgraph_localization import SubgraphIsomorphismSolver


class MetricEvaluator:
    def __init__(self, A):
        super().__init__()
        self.A = A

    def iou(self,v_np, v_gt):
        return jaccard_score(v_np,v_gt,average=None)

    def accuracy(self, v_np, v_gt):
        v_clustered, v_, e_clustered, e_ = \
            self._arrange_data(v_np=v_np, v_gt=v_gt)

        return accuracy_score(y_true=v_, y_pred=v_clustered), \
               accuracy_score(y_true=e_, y_pred=e_clustered)

    def balanced_accuracy(self, v_np, v_gt, E_np=None, E_gt=None):
        v_clustered, v_, e_clustered, e_ = \
            self._arrange_data(v_np=v_np, v_gt=v_gt)

        return balanced_accuracy_score(y_true=v_, y_pred=v_clustered), \
               balanced_accuracy_score(y_true=e_, y_pred=e_clustered)

    def _arrange_data(self, v_np, v_gt):
        v_ = v_np - np.min(v_np)
        v_ = v_ / np.max(v_)
        v_clustered, centroids = kmeans1d.cluster(v_, k=2)
        E_clustered, S_clustered = \
            SubgraphIsomorphismSolver.E_from_v(torch.tensor(v_clustered), self.A)

        v_ = v_gt.astype(int) - np.min(v_gt.astype(int))
        v_ = v_ / np.max(v_)

        E_gt, S_gt = SubgraphIsomorphismSolver.E_from_v(torch.tensor(v_gt), self.A)
        S_ = S_gt.numpy()
        e_clustered = S_clustered[np.triu_indices(len(v_np))]
        e_ = S_[np.triu_indices(len(v_np))]
        return v_clustered, v_, e_clustered, e_

    def evaluate(self, v_np, v_gt):
        metrics = {"accuracy": self.accuracy(v_np, v_gt),
                   "balanced accuracy": self.balanced_accuracy(v_np, v_gt)}
        return metrics

    @staticmethod
    def print(metrics):
        keys = metrics[0].keys()
        for key in keys:
            print(f"v {key} = {[metric[key][0] for metric in metrics]}")
            print(f"E {key} = {[metric[key][1] for metric in metrics]}")

