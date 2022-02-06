import numpy as np
import torch
import torch.nn.functional as F
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
        layers_sizes = params.get('layers_sizes', [500, 250])
        layers_sizes.insert(0, in_dim)   # Insert the input dimension

        # Add deep layers
        modules = []
        for i in range(len(layers_sizes) - 1):
            modules.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1], bias=True))
            # modules.append(nn.ReLU())
            # modules.append(nn.Identity())

        # Add last layer for Z
        modules.append(nn.Linear(layers_sizes[-1], latent_dim, bias=True))
        # modules.append(nn.ReLU())
        # modules.append(nn.Identity())
        
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
        layers_sizes = params.get('layers_sizes', [250, 500])
        layers_sizes.append(out_dim)   # append the output dimension

        modules = []
        # Add last layer for Z
        modules.append(nn.Linear(latent_dim, layers_sizes[0], bias=True))
        # modules.append(nn.ReLU())
        modules.append(nn.Sigmoid())

        # Add deep layers
        for i in range(len(layers_sizes) - 1):
            modules.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1], bias=True))
            # modules.append(nn.ReLU())
            modules.append(nn.Sigmoid())

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
    def __init__(self, n_dim, params):
        super().__init__()
        self.p_dims = params.get('p_dims', [250, 500]).copy()
        self.p_dims.append(n_dim)
        self.activation_func = params.get('activation_func', 'tanh')
        dropout = params.get('dropout', 0.5)
        q_dims = params.get('q_dims', None)

        if q_dims:
            assert q_dims[0] == self.p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == self.p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = self.p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                if self.activation_func == 'tanh':
                    h = torch.tanh(h)
                elif self.activation_func == 'relu':
                    h = torch.relu(h)
                elif self.activation_func == 'selu':
                    h = torch.selu(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                if self.activation_func == 'tanh':
                    h = torch.tanh(h)
                elif self.activation_func == 'relu':
                    h = torch.relu(h)
                elif self.activation_func == 'selu':
                    h = torch.selu(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


def get_model(model_name, params, dl_train):
    """
    Instantiate the proper model based on the model_name parameter. 
    Use the needed hyperparameters from params.
    Also, extract the needed data dimensions for building the models.
    """
    model = None

    if model_name == 'MF':
        num_users = dl_train.dataset.num_users
        num_items = dl_train.dataset.num_items
        model = MF(num_users=num_users, num_items=num_items, params=params)
    elif model_name == 'AutoRec':
        n_dim = dl_train.dataset.__getitem__(1).shape[0]
        linear_encoder = EncoderLinear(in_dim=n_dim, params=params)
        linear_decoder = DecoderLinear(out_dim=n_dim, params=params)
        model = AutoRec(encoder=linear_encoder, decoder=linear_decoder)
    elif model_name == 'VAE':
        n_dim = dl_train.dataset.__getitem__(1).shape[0]
        model = VAE(n_dim=n_dim, params=params)
    return model