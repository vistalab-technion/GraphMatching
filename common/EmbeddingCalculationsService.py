import torch
import torch.nn.functional as F

from common.PlotService import plot_heatmap_basic


def pairwise_l2_distance(x):
    original_device = x.device
    tmp_device = "cpu"
    x = x.to(tmp_device)

    n = x.size(0)
    if len(x.shape) == 1:
        x1 = x.unsqueeze(1).expand(n, n)
        x2 = x.unsqueeze(0).expand(n, n)
        dist = torch.pow(x1 - x2, 2)
    else:
        d = x.size(1)
        x1 = x.unsqueeze(1).expand(n, n, d)
        x2 = x.unsqueeze(0).expand(n, n, d)
        dist = torch.pow(x1 - x2, 2).sum(2)

    return dist.to(original_device)

def cosine_pairwise(x, threshold):
    x = x.float().unsqueeze(0).permute((1, 2, 0))
    # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
    cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))[0]

    cos_sim_pairwise = torch.maximum(cos_sim_pairwise, threshold * torch.ones(cos_sim_pairwise.shape, device=x.device))
    above_threshold_indications = torch.gt(cos_sim_pairwise, threshold)
    cos_sim_pairwise = cos_sim_pairwise * above_threshold_indications

    return cos_sim_pairwise


def calculate_energy_based_hidden_rep(hidden_rep, threshold=0.5):
    cosine_similarity = cosine_pairwise(hidden_rep, threshold)
    return 1 - cosine_similarity


def show_distance_matrix(distances, title="GraphEmbeddingsDistancesMatrix"):
    #distances = torch.pow(torch.cdist(graph_embeddings, graph_embeddings), 2)
    plot_heatmap_basic(distances.detach().cpu().numpy(), title)