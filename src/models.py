import torch
from torch import nn


class MF(nn.Module):
    """
    Matrix Factorizaion model implementation
    """
    def __init__(self, num_users, num_items, params):
        super().__init__()
        # Get relevant hyperparameters from the params dict
        latent_dim = params['latent_dim']

        # Initialize embedding layers for the users and for the items
        self.embedding_user = torch.nn.Embedding(num_embeddings=num_users+1, embedding_dim=latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=num_items+1, embedding_dim=latent_dim)

    def forward(self, user_indices, item_indices):
        # Get the user and item vector using the embedding layers
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        # Calculate the rating for the user-item combination
        output = (user_embedding * item_embedding).sum(1)
        return output


class EncoderLinear(nn.Module):
    """
    Encoder implementation (can be used for both, AutoRec and VAE)
    """
    def __init__(self, in_dim, params):
        super().__init__()
        # Get relevant hyperparameters from the params dict
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
        
        # Generate the layers sequence foe easier implementation
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


class DecoderLinear(nn.Module):
    """
    Decoder implementation (can be used for both, AutoRec and VAE)
    """
    def __init__(self, out_dim, params):
        super().__init__()
        # Get relevant hyperparameters from the params dict
        latent_dim = params.get('latent_dim', 10)
        layers_sizes = params.get('layers_sizes', [256, 512])
        layers_sizes.append(out_dim)   # append the output dimension

        modules = []
        # Add last layer for Z
        modules.append(nn.Linear(latent_dim, layers_sizes[0], bias=True))
        modules.append(nn.ReLU())

        # Add deep layers
        for i in range(len(layers_sizes) - 1):
            modules.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1], bias=True))
            modules.append(nn.ReLU())

        # Generate the layers sequence foe easier implementation
        self.seq = nn.Sequential(*modules)

    def forward(self, z):
        return self.seq(z)


class AutoRec(nn.Module):
    """
    AutoRec model implementation
    """
    def __init__(self, encoder, decoder):
        super().__init__()

        # Use external implementation of encoder & decoder (actually, using the above EncoderLinear & DecoderLinear)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VAE(nn.Module):
    """
    VAE model implementation
    """
    def __init__(self):
        super().__init__()

    def forward(self):
        pass