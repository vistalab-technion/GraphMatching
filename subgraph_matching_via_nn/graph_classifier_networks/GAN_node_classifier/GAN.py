import torch
from torch.utils.data import DataLoader
import pandas as pd
from numpy import random

from bgan_pytorch.bgan.losses import multinomial_bgan_loss, binary_bgan_loss
from bgan_pytorch.bgan.model import Model
from bgan_pytorch.bgan.utils import apply_spectral_norm, create_result_dir, init_weights, get_activation_by_name
from subgraph_matching_via_nn.graph_classifier_networks.GAN_node_classifier.dataset import gan_dataset
from subgraph_matching_via_nn.graph_classifier_networks.GAN_node_classifier.discriminator_model import Discriminator
from subgraph_matching_via_nn.graph_classifier_networks.GAN_node_classifier.gan_trainer import GANTrainer
from subgraph_matching_via_nn.graph_classifier_networks.GAN_node_classifier.generator_model import Generator
from subgraph_matching_via_nn.utils.utils import TORCH_DTYPE


def generate_example_data(num_features, device, dist_type):
    if dist_type == "uniform":
        X = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(5_000, num_features))
    elif dist_type == "normal":
        X = random.normal(size=(5_000, num_features))
    else:
        raise NotImplementedError(dist_type)

    data = pd.DataFrame(X)

    X_normalized = torch.FloatTensor((X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * 2 - 1).to(dtype=TORCH_DTYPE,
                                                                                                       device=device)
    real_data = X_normalized
    # num_features = 1 if len(real_data.shape) < 2 else real_data.shape[1]
    # if num_features == 1:
    #     real_data = real_data.reshape(-1, 1)
    return data, real_data

def bggan_train(run_name, device,
          G, D, dataset, activation,
          use_spectral_norm,
          d_lr,
          g_lr,
          batch_size,
          n_sample,
          n_mc_samples,
          num_workers,
          epochs,
          log_every,
          sample_every, dtype):
    result_dir, sample_dir, checkpoint_dir = create_result_dir(run_name)

    if dataset.num_colors <= 2:
        # dim = 1
        loss_f = binary_bgan_loss
    else:
        # dim = dataset.num_colors
        loss_f = multinomial_bgan_loss

    init_weights(G)
    init_weights(D)

    if use_spectral_norm:
        apply_spectral_norm(G)
        apply_spectral_norm(D)

    G_opt = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
    D_opt = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))

    model = Model(
        G, D,
        G_opt, D_opt,
        loss_f=loss_f,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        sample_folder=None,
        checkpoint_folder=checkpoint_dir,
        n_sample=n_sample,
        n_mc_samples=n_mc_samples,
        num_workers=num_workers,
        dtype=dtype
    )
    model.train(epochs, log_every, sample_every)


if __name__ == "__main__":

    # general configuration
    device = "cuda"
    dtype = TORCH_DTYPE

    # generating example DATA
    num_features = 2 #1 #2 #2
    dist_type = "uniform" # "normal
    data, real_data = generate_example_data(num_features=num_features, device=device, dist_type=dist_type)
    data = pd.DataFrame(real_data) #TODO? keep this row?

    # Defining training parameters
    batch_size = 512
    num_epochs = 50 #1_000# 500 #150
    lr = 0.0002
    latent_dim = 20
    noise_dim = latent_dim

    # MODEL INITIALIZATION
    activation_name = 'elu'
    activation = get_activation_by_name(activation_name)
    generator = Generator(noise_dim, num_features, device=device, activation=activation)
    discriminator = Discriminator(num_features, device=device, activation=activation)

    # Create a dataloader of the dataset
    dataset = gan_dataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    use_simple_gan = False
    run_name = "DEMO_GAN"
    num_workers = 0 # for faster performance -> try on lab machine

    if use_simple_gan:
        # train models
        GANTrainer.train_gan(discriminator, generator, dataloader, num_epochs, lr, dtype, device)

    else:
        bggan_train(run_name, device, generator, discriminator, dataset, activation=activation, use_spectral_norm=True, d_lr=lr, g_lr=lr, batch_size=batch_size,
                    n_sample=min(16, len(data)), n_mc_samples=20, num_workers=num_workers, epochs=num_epochs, log_every=20, sample_every=num_epochs, dtype=dtype)

    ## Evaluating and visualising the results
    GANTrainer.evaluate_results(generator, real_data, noise_dim, dtype, device=device)
