import torch
import torch.nn as nn
import torch.nn.functional as F


from layers import GraphConvolution
from utils import get_subsampler

class GAE(nn.Module):
    """Graph Auto Encoder (see: https://arxiv.org/abs/1611.07308) - Probabilistic Version"""

    def __init__(self, data, n_hidden, n_latent, dropout):
        super().__init__()

        # Data
        self.x = data['features']
        self.adj_norm = data['adj_norm']
        self.adj_labels = data['adj_labels']    

        # Dimensions
        N, D = data['features'].shape
        self.n_samples = N
        self.n_edges = self.adj_labels.sum()
        self.n_subsample = 2 * self.n_edges
        self.input_dim = D
        self.n_hidden = n_hidden
        self.n_latent = n_latent

        # Parameters
        self.pos_weight = float(N * N - self.n_edges) / self.n_edges
        self.norm = float(N * N) / ((N * N - self.n_edges) * 2)

        self.gc1 = GraphConvolution(self.input_dim, self.n_hidden)
        self.gc2 = GraphConvolution(self.n_hidden, self.n_latent)
        self.dropout = dropout
    def encode_graph(self, x, adj):

        # Perform the encoding stage using a two layer GCN
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x

    def forward(self, x, adj):

        # Encoder
        x = self.encode_graph(x, adj)
        # Decoder
        x = F.dropout(x, self.dropout, training=self.training)
        adj_hat = torch.mm(x, x.t())

        return adj_hat

    def get_embeddings(self, x, adj):

        return self.encode_graph(x, adj)

class GVAE(nn.Module):
    """Graph Auto Encoder (see: https://arxiv.org/abs/1611.07308) - Variational Version without the use of pyro"""

    def __init__(self, data, n_hidden, n_latent, dropout):
        super().__init__()

        # Data
        self.x = data['features']
        self.adj_norm = data['adj_norm']
        self.adj_labels = data['adj_labels']    

        # Dimensions
        N, D = data['features'].shape
        self.n_samples = N
        self.n_edges = self.adj_labels.sum()
        self.n_subsample = 2 * self.n_edges
        self.input_dim = D
        self.n_hidden = n_hidden
        self.n_latent = n_latent

        # Parameters
        self.pos_weight = float(N * N - self.n_edges) / self.n_edges
        self.norm = float(N * N) / ((N * N - self.n_edges) * 2)

        self.gc1 = GraphConvolution(self.input_dim, self.n_hidden)
        self.gc2_mu = GraphConvolution(self.n_hidden, self.n_latent)
        self.gc2_sig = GraphConvolution(self.n_hidden, self.n_latent)
        self.dropout = dropout

    def encode_graph(self, x, adj):
        # First layer shared between mu/sig layers
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        self.mu = self.gc2_mu(x, adj)
        self.log_sig = self.gc2_sig(x, adj)

        self.z = self.mu + torch.randn(self.n_samples ,self.n_latent) * torch.exp(self.log_sig)

        return self.z

    def decode_graph(self, x):
        # Here the reconstruction is based upon 
        adj_hat = torch.mm(x, x.t())

        return adj_hat

    def get_embeddings(self, x, adj):

        return self.encode_graph(x, adj)

    def forward(self, x, adj):
        # Encode and then decode the graph
        x_hat = self.encode_graph(x, adj)
        adj_hat = self.decode_graph(x_hat)

        return adj_hat
