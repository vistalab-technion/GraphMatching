import torch
from torch import nn
from torch.nn import functional as F


class MarginLoss(nn.Module):

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, distances, pos_labels, neg_labels):
        pos_loss, neg_loss = self.get_loss(distances, pos_labels, neg_labels)

        # self.print_loss_statistics(distances, pos_labels, neg_labels)

        return pos_loss + neg_loss

    def get_loss(self, distances, pos_labels, neg_labels):
        pos_loss = pos_labels.type(dtype=torch.float) * distances
        neg_loss = neg_labels * (torch.clamp(self.margin - distances, min=0.0) ** 2) / 2
        # print(f'pos loss: {pos_loss.sum()}\t neg loss: {neg_loss.sum()}')

        return pos_loss, neg_loss

    def print_loss_statistics(self, distances, pos_labels, neg_labels):
        distances = distances.detach()
        pos_loss, neg_loss = self.get_loss(distances, pos_labels, neg_labels)

        inf_diag = torch.ones(distances.shape[0], device=distances.device, requires_grad=False) * float('inf')
        inf_diag = torch.diag(inf_diag)
        only_negative_pairs_distances = (distances + inf_diag)
        min_neg_value = torch.min(only_negative_pairs_distances)

        below_margin_count = torch.sum(only_negative_pairs_distances < self.margin)
        zero_distance_negative_pairs = torch.sum(only_negative_pairs_distances == 0)

        non_zero_count = torch.count_nonzero(pos_loss)
        print(f'margin = {self.margin}, min_neg_value = {min_neg_value}, below_margin_count = {below_margin_count}, 0_margin_count = {zero_distance_negative_pairs}, non_zero={non_zero_count}')
        #show_distance_matrix(euclidean_distances)
        return below_margin_count, torch.sum(pos_loss + neg_loss)


def calc_contrastive_loss(pairs, label, margin, device):
    lossCriterion = MarginLoss(margin)
    first_element_copies = pairs[0, :].repeat((pairs.shape[0], 1))
    euclidean_distance = F.pairwise_distance(first_element_copies, pairs)
    loss = lossCriterion(euclidean_distance, label)

    #losses = (label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    # distance = torch.pow(torch.cdist(pairs[0].reshape(1, -1), pairs[1:]), 2).reshape(-1)
    # losses = label * torch.pow(distance, 2) + (1-label) * torch.pow(torch.max(torch.zeros(distance.shape[0], device=device), margin - distance), 2)

    #loss = pos_loss + neg_loss
    return loss