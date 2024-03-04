from typing import Sequence
import numpy as np
from sklearn.metrics import roc_curve, recall_score, f1_score, auc, accuracy_score, \
    precision_score, confusion_matrix, precision_recall_curve
import pandas as pd
import seaborn as sns


def compute_metrics(y_true: Sequence[int],
                    y_pred: Sequence[int],
                    y_proba: Sequence[float] = None):
    """

    :param y_true: ground truth labels of
    :param y_pred: predicted labels
    :param y_proba: probabilities of belonging to each class
    :return:
    """
    # Calculating metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_metric = precision_score(y_true, y_pred)
    recall_metric = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = auc(roc_curve(y_true, y_proba)[0],
                  roc_curve(y_true, y_proba)[1]) if y_proba is not None else 'N/A'

    # Creating a summary table
    summary_table = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
        'Value': [accuracy, precision_metric, recall_metric, f1, roc_auc]
    })
    return summary_table


def plot_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                 axes=None, y_proba: Sequence[float] = None):
    """
    :param y_true: ground truth labels of
    :param y_pred: predicted labels
    :param axes: 1x3 axes to plot on (if y_proba is not None)
    :param y_proba: probabilities of belonging to each class
    :return: best threshold of the ROC curve (i.e. best trade-off between precision
     and recall)
    """
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Initialize the best threshold variables
    best_threshold = None
    min_diff = float('inf')

    # Plotting Confusion Matrix
    sns.heatmap(conf_matrix, annot=False, fmt='g', cmap='Blues', cbar=False, ax=axes[0])
    labels = ["TN", "FP", "FN", "TP"]
    counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    percentages = ["{0:.2%}".format(value) for value in
                   conf_matrix.flatten() / np.sum(conf_matrix)]
    label_texts = [f"{label}\n{count}\n{percentage}" for label, count, percentage in
                   zip(labels, counts, percentages)]
    label_texts = np.asarray(label_texts).reshape(2, 2)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            axes[0].text(j + 0.5, i + 0.5, label_texts[i, j],
                         horizontalalignment='center',
                         verticalalignment='center',
                         color="black")

    if y_proba is not None:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
        for p, r, t in zip(precision, recall, pr_thresholds):
            diff = abs(p - r)
            if diff < min_diff:
                min_diff = diff
                best_threshold = t

        # Plotting ROC Curve
        axes[1].plot(fpr, tpr, color='blue', lw=2,
                     label=f'ROC curve (area = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")

        # Plotting Precision-Recall Curve
        axes[2].plot(recall, precision, color='green', lw=2)
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title(f'Precision-Recall, t_best = {best_threshold:.3f}')
    return best_threshold


def evaluate_binary_classifier(y_true: Sequence[int], y_pred: Sequence[int],
                               fig=None, axes=None, y_proba: Sequence[float] = None):
    """
    :param y_true: ground truth labels of
    :param y_pred: predicted labels
    :param fig: fig to plot on (if y_proba is not None)
    :param axes: 1x3 axes to plot on (if y_proba is not None)
    :param y_proba: probabilities of belonging to each class
    :return:
    """
    summary_table = compute_metrics(y_true=y_true,
                                    y_pred=y_pred,
                                    y_proba=y_proba)
    if fig is not None and axes is not None:
        plot_metrics(y_true=y_true,
                     y_pred=y_pred,
                     axes=axes,
                     y_proba=y_proba)
    return summary_table
