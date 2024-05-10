import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.init as init
import numpy as np
import seaborn as sns


class GANTrainer:

    @staticmethod
    def __init_models_weights(generator, discriminator, pretrained=False):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        if pretrained:
            # pre_dict = torch.load('pretrained_model.pth')
            # generator.load_state_dict(pre_dict['generator'])
            # discriminator.load_state_dict(pre_dict['discriminator'])
            raise NotImplementedError("Pretrained option currently not supported")
        else:
            # Apply weight initialization
            generator = generator.apply(weights_init)
            discriminator = discriminator.apply(weights_init)

        return generator, discriminator

    @staticmethod
    def train_gan(discriminator, generator, dataloader, num_epochs, lr, dtype, device):
        generator, discriminator = GANTrainer.__init_models_weights(generator, discriminator)

        # LOSS FUNCTION AND OPTIMIZERS
        criterion = nn.BCELoss()
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

        for epoch in range(num_epochs):
            for batch in dataloader:
                real_data_batch = batch['input'].to(device=device)
                batch_size = real_data_batch.shape[0]
                # Train discriminator on real data
                real_labels = torch.FloatTensor(np.random.uniform(0.9, 1.0, (batch_size, 1))).to(dtype=dtype,
                                                                                                 device=device)
                disc_optimizer.zero_grad()

                discriminator.train()
                output_real = discriminator(real_data_batch)
                loss_real = criterion(output_real, real_labels)
                # loss_real.backward()
                d_loss = loss_real

                # Train discriminator on generated data
                fake_labels = torch.FloatTensor(np.random.uniform(0, 0.1, (batch_size, 1))).to(dtype=dtype,
                                                                                               device=device)
                noise = torch.FloatTensor(np.random.normal(0, 1, (batch_size, generator.latent_dimension_size))).to(
                    dtype=dtype, device=device)

                generator.eval()
                with torch.no_grad():
                    generated_data = generator(noise)

                output_fake = discriminator(generated_data.detach())
                loss_fake = criterion(output_fake, fake_labels)
                d_loss += loss_fake

                d_loss.backward()

                disc_optimizer.step()

                # Train generator
                generator.train()
                generated_data = generator(noise)

                valid_labels = torch.FloatTensor(np.random.uniform(0.9, 1.0, (batch_size, 1))).to(dtype=dtype,
                                                                                                  device=device)
                gen_optimizer.zero_grad()

                discriminator.eval()  # eval but we still need gradients
                output_g = discriminator(generated_data)
                loss_g = criterion(output_g, valid_labels)
                loss_g.backward()
                gen_optimizer.step()

            # Print progress
            print(
                f"Epoch {epoch}, D Loss Real: {loss_real.item()}, D Loss Fake: {loss_fake.item()}, G Loss: {loss_g.item()}")

    @staticmethod
    def evaluate_results(generator, real_data, noise_dim, dtype, device):
        num_features = real_data.shape[1]

        # Generate synthetic data
        synthetic_data = generator(
            torch.FloatTensor(np.random.normal(0, 1, (real_data.shape[0], noise_dim))).to(dtype=dtype, device=device))

        # Plot the results
        fig, axs = plt.subplots(num_features, figsize=(12, 8))
        fig.suptitle('Real and Synthetic Data Distributions', fontsize=16)

        if num_features == 1:
            axs = [axs]

        for i in range(num_features):
            # for j in range(3):
            sns.histplot(synthetic_data[:, i].reshape(-1, 1).detach().cpu().numpy(), bins=50, alpha=0.5,
                         label='Synthetic Data', ax=axs[i], color='blue')
            sns.histplot(real_data[:, i].reshape(-1, 1).cpu().numpy(), bins=50, alpha=0.5, label='Real Data', ax=axs[i],
                         color='orange')
            axs[i].set_title(f'Parameter {i + 1}', fontsize=12)
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Frequency')
            axs[i].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Create a grid of subplots
        fig, axs = plt.subplots(num_features, figsize=(15, 10))
        fig.suptitle('Comparison of Real and Synthetic Data', fontsize=16)

        if num_features == 1:
            axs = [axs]

        # Scatter plots for each parameter
        for i in range(num_features):
            # for j in range(3):
            param_index = i  # * 3 + j
            sns.scatterplot(real_data[:, i].detach().cpu().numpy(), real_data[:, param_index].detach().cpu().numpy(),
                            label='Real Data', alpha=0.5, ax=axs[i])
            sns.scatterplot(synthetic_data[:, i].detach().cpu().numpy(),
                            synthetic_data[:, param_index].detach().cpu().numpy(), label='Generated Data', alpha=0.5,
                            ax=axs[i])
            axs[i].set_title(f"Parameter {param_index + 1}", fontsize=12)
            axs[i].set_xlabel(f'Real Data - {param_index + 1}')
            axs[i].set_ylabel(f'Real Data - {param_index + 1}')
            axs[i].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
