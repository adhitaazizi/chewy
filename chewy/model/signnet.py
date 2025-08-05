from typing import List

import torch
import torch.nn as nn
from chewy.model.components.phi import PhiModule
from chewy.model.components.rho import RhoModule


class SignNet(nn.Module):
    """
    SignNet for processing Laplacian eigenvectors, based on the paper
    "SIGN AND BASIS INVARIANT NETWORKS FOR SPECTRAL GRAPH REPRESENTATION LEARNING".
    """
    def __init__(self, in_channels, phi_out_dim, num_phi_layers, rho_out_dim, num_rho_layers):
        super().__init__()

        self.phi = PhiModule(in_channels, phi_out_dim, num_phi_layers)
        self.rho = RhoModule(phi_out_dim, rho_out_dim, num_rho_layers, heads=4, dropout=0.0)

    def forward(self, eigvecs, edge_index):
        # Process each eigenvector as a feature on the graph
        num_nodes, k = eigvecs.shape
        eigvecs_reshaped = eigvecs.T.reshape(num_nodes * k, 1)
        edge_index_batch = edge_index.repeat(1, k) + torch.arange(k, device=edge_index.device).repeat_interleave(edge_index.size(1)) * num_nodes

        # phi(v) + phi(-v)
        pe_pos = self.phi(eigvecs_reshaped, edge_index_batch)
        pe_neg = self.phi(-eigvecs_reshaped, edge_index_batch)
        pe = pe_pos + pe_neg

        pe = pe.view(k, num_nodes, -1).permute(1,0,2)
        pe = torch.sum(pe, dim=1)
        
        # Apply rho network for final encoding
        pe = self.rho(pe)
        
        return pe