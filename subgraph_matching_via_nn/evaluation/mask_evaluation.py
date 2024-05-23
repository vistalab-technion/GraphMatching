from metrics.accuracy_metrics import evaluate_binary_classifier


def evaluate_mask_performance(w_bin, gt_node_distribution_processed):
    return evaluate_binary_classifier(y_pred =
                               (w_bin/max(w_bin)).astype(bool),
                               y_true= (gt_node_distribution_processed/max(gt_node_distribution_processed)).squeeze().numpy().astype(bool))
