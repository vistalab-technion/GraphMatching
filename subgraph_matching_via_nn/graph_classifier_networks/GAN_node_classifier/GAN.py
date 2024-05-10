import torch
from torch.utils.data import DataLoader
import pandas as pd
from numpy import random

from subgraph_matching_via_nn.graph_classifier_networks.GAN_node_classifier.dataset import gan_dataset
from subgraph_matching_via_nn.graph_classifier_networks.GAN_node_classifier.discriminator_model import Discriminator
from subgraph_matching_via_nn.graph_classifier_networks.GAN_node_classifier.gan_trainer import GANTrainer
from subgraph_matching_via_nn.graph_classifier_networks.GAN_node_classifier.generator_model import Generator
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


def generate_example_data(num_features, device):
    # X = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(5_000, num_features))
    X = random.normal(size=(5_000, num_features))

    data = pd.DataFrame(X)

    X_normalized = torch.FloatTensor((X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * 2 - 1).to(dtype=TORCH_DTYPE,
                                                                                                       device=device)
    real_data = X_normalized
    # num_features = 1 if len(real_data.shape) < 2 else real_data.shape[1]
    # if num_features == 1:
    #     real_data = real_data.reshape(-1, 1)
    return data, real_data


if __name__ == "__main__":

    # general configuration
    device = "cuda"
    dtype = TORCH_DTYPE

    # generating example DATA
    num_features = 1 #2
    data, real_data = generate_example_data(num_features=num_features, device=device)
    data = pd.DataFrame(real_data) #TODO? keep this row?

    # Defining training parameters
    batch_size = 512
    num_epochs = 10 #1_000# 500 #150
    lr = 0.0002
    latent_dim = 20
    noise_dim = latent_dim

    # MODEL INITIALIZATION
    generator = Generator(noise_dim, num_features, device=device)
    discriminator = Discriminator(num_features, device=device)

    # Create a dataloader of the dataset
    dataset = gan_dataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # train models
    GANTrainer.train_gan(discriminator, generator, dataloader, num_epochs, lr, dtype, device)

    ## Evaluating and visualising the results
    GANTrainer.evaluate_results(generator, real_data, noise_dim, dtype, device=device)
