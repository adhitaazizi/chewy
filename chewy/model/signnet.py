from typing import List

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GATConv
from torch.nn import Sequential, Linear, ReLU
from chewy.model.components.phi import PhiModule
from chewy.model.components.rho import RhoModule


class SignNet(nn.Module):
    """
    SignNet for processing Laplacian eigenvectors, based on the paper
    "SIGN AND BASIS INVARIANT NETWORKS FOR SPECTRAL GRAPH REPRESENTATION LEARNING".
    """
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int, num_layers_phi: int = 4, num_layers_rho: int = 4):
        super().__init__()

        # phi network to process each eigenvector
        self.phi = PhiModule(in_channels, hidden_dim, num_layers_phi)

        # rho network to aggregate the processed eigenvectors
        self.rho = RhoModule(hidden_dim, out_dim, num_layers_rho, heads=4, dropout=0.0)

    def forward(self, data):
        # eigvecs: (num_nodes, num_eigenvectors)
        
        # The paper suggests processing each eigenvector independently.
        # We can achieve this by treating the eigenvectors as a batch.
        # Reshape for independent processing: (num_nodes * num_eigenvectors, 1)
        num_nodes, k = eigvecs.shape
        reshaped_eigvecs = eigvecs.T.reshape(-1, 1)

        # Apply phi to both v and -v
        pe = self.phi(reshaped_eigvecs) + self.phi(-reshaped_eigvecs) # [cite: 125]
        
        # Reshape back to (num_nodes, k, hidden_dim)
        pe = pe.view(k, num_nodes, -1).permute(1,0,2)
        
        # Sum over the eigenvectors for each node before rho
        pe = torch.sum(pe, dim=1)
        
        # Apply rho to get the final positional encoding
        pe = self.rho(pe) # [cite: 200]
        
        return pe