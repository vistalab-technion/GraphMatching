import torch
from torch import stack, Tensor

POS_LABEL = 1
NEG_LABEL = -1


def get_euclidean_distance(z1, z2, embedding_function):
    z1_embedded = embedding_function(stack(z1).unsqueeze(1))
    z2_embedded = embedding_function(stack(z2).unsqueeze(1))

    return (torch.abs(z1_embedded - z2_embedded)).reshape(z1_embedded.shape[0], -1).sum(dim=1)


def stable_inverse(x, epsilon):
    return 1 / (x + epsilon)


def inv_sum_reciprocal(x, y, epsilon1=1e-6):
    return stable_inverse(stable_inverse(x+y, epsilon1), 0)


# need to compute gradient on all pairs
# the label is obtained according to comparison to ground truth, i.e.,
# if it is a local minima but not the one we want, it gets negative label.
def get_kernel_training_loss(grad_distances, euclidean_distances, pos_labels, neg_labels, kernel_loss_fn, tol=1e-7):
    """"
    label - positive or negative per pair
    margin - some positive number
    """

    device = pos_labels.device

    if grad_distances is None:
        grad_distances = float("Inf") * torch.ones(size=pos_labels.shape, device=device, requires_grad=False)
    if euclidean_distances is None:
        euclidean_distances = float("Inf") * torch.ones(size=pos_labels.shape, device=device, requires_grad=False)

    total_distance_sqr = combine_grad_with_euclidean_distances(grad_distances, euclidean_distances, tol)
    loss = kernel_loss_fn(total_distance_sqr, pos_labels.reshape(-1), neg_labels.reshape(-1))
    return loss.sum()


# the motivation is making sure that future inference loss won't converge into current negative examples instances
# (e.g. graphs/patches), by making sure
# either do not impose local minimum (via grad_distances)
# or they are not global minimum - don't longer have a similar embedding to their GT pair instance's embedding (via euclidean_distances)
def combine_grad_with_euclidean_distances(grad_distances, euclidean_distances, tol=1e-7):
    if isinstance(euclidean_distances, list):
        euclidean_distances = stack(euclidean_distances)

    if isinstance(grad_distances, list):
        grad_distances = stack(grad_distances)

    euclidean_distances = euclidean_distances.reshape(-1)
    grad_distances = grad_distances.reshape(-1)

    euclidean_distances = torch.nan_to_num(euclidean_distances, nan=float("Inf"))
    grad_distances = torch.nan_to_num(grad_distances, nan=float("Inf"))

    total_distance_sqr = inv_sum_reciprocal(grad_distances, euclidean_distances, epsilon1=tol / 100)
    return total_distance_sqr


def l2_norm_loss(z1, z2):
    return torch.norm(z1 - z2) ** 2


def is_local_minimum(params_grad, gradient_diff_threshold=5e-3):
    gradient_distance = torch.norm(params_grad).item()
    has_converged = gradient_distance < gradient_diff_threshold
    return has_converged, gradient_distance


# calculate euclidean-grad distance for single graph pairs batch
def get_pairs_batch_aggregated_distance(graph_similarity_loss_function, emb_dist, grad_distance, is_negative_example) \
        -> Tensor:
    if type(is_negative_example) is bool:
        batch_size = emb_dist.shape[0]
        pos_labels = torch.ones(batch_size, requires_grad=False, device=emb_dist.device)
        if is_negative_example:
            pos_labels = 0 * pos_labels
        neg_labels = 1 - pos_labels
    else:
        neg_labels = torch.stack(is_negative_example).to(device=emb_dist.device).requires_grad_(False).long()
        pos_labels = 1 - neg_labels
    return get_kernel_training_loss(grad_distance, emb_dist, pos_labels, neg_labels, graph_similarity_loss_function)
