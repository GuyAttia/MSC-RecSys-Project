from turtle import forward
import torch
from torch import nn


class MF(nn.Module):
  def __init__(self, num_users, num_items, params):
    super().__init__()
    latent_dim = params['latent_dim']
    self.embedding_user = torch.nn.Embedding(num_embeddings=num_users+1, embedding_dim=latent_dim)
    self.embedding_item = torch.nn.Embedding(num_embeddings=num_items+1, embedding_dim=latent_dim)

  def forward(self, user_indices, item_indices):
    user_embedding = self.embedding_user(user_indices)
    item_embedding = self.embedding_item(item_indices)

    output = (user_embedding * item_embedding).sum(1)
    return output


class EncoderLinear(nn.Module):
    def __init__(self, in_dim, params):
        super().__init__()
        latent_dim = params.get('latent_dim', 10)
        layers_sizes = params.get('layers_sizes', [512, 256])
        layers_sizes.insert(0, in_dim)   # Insert the input dimension

        # Add deep layers
        modules = []
        for i in range(len(layers_sizes) - 1):
            modules.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1], bias=True))
            modules.append(nn.ReLU())

        # Add last layer for Z
        modules.append(nn.Linear(layers_sizes[-1], latent_dim, bias=True))
        modules.append(nn.ReLU())

        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


class DecoderLinear(nn.Module):
    def __init__(self, out_dim, params):
        super().__init__()
        latent_dim = params.get('latent_dim', 10)
        layers_sizes = params.get('layers_sizes', [256, 256])
        layers_sizes.append(out_dim)   # append the output dimension

        modules = []

        # Add last layer for Z
        modules.append(nn.Linear(latent_dim, layers_sizes[0], bias=True))
        modules.append(nn.ReLU())

        # Add deep layers
        for i in range(len(layers_sizes) - 1):
            modules.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1], bias=True))
            modules.append(nn.ReLU())

        self.seq = nn.Sequential(*modules)

    def forward(self, z):
        return self.seq(z)


class AutoRec(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass