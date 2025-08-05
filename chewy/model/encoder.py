from typing import List

import torch
import torch.nn as nn
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from chewy.model.module import GATModule
    
class ChewyEncoder(nn.Module):
    """
    DeeperGCN-style GAT Encoder with SignNet for positional encoding.
    """
    def __init__(
        self,
        input_dim: int,
        dim_list: List[int],
        heads: int = 1,
        dropout: float = 0.0,
        pe_k: int = 8,
        pe_out_dim: int = 32, # Output dimension for SignNet
        use_pe: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.dim_list = dim_list
        self.use_pe = use_pe
        self.pe_k = pe_k
        
        if self.use_pe:
            self.sign_net = SignNet(in_channels=input_dim, hidden_dim=64, out_dim=pe_out_dim)
            effective_input_dim = input_dim + pe_out_dim
        else:
            effective_input_dim = input_dim

        # Build encoder layers
        self.layers = nn.ModuleList()
        
        # First layer: effective_input_dim -> dim_list[0]
        self.layers.append(
            GATModule(
                in_channels=effective_input_dim,
                out_channels=dim_list[0],
                heads=heads,
                dropout=dropout,
            )
        )
        
        # Intermediate layers
        for i in range(len(dim_list) - 1):
            self.layers.append(
                GATModule(
                    in_channels=dim_list[i] * heads,
                    out_channels=dim_list[i + 1],
                    heads=heads,
                    dropout=dropout,
                )
            )
        
        # Final normalization
        self.final_norm = nn.LayerNorm(dim_list[-1] * heads)
    
    def forward(self, data):
        x = data.x  
        edge_index = data.edge_index
        
        if self.use_pe:
            # Assuming Laplacian eigenvectors are stored in data.lap_eigvec
            # You would typically compute this using a transform, e.g., AddLaplacianEigenvectorPE
            pe = self.sign_net(data.lap_eigvec[:, :self.pe_k])
            x = torch.cat([x, pe], dim=-1) # [cite: 92]
        
        # Pass through all layers
        for layer in self.layers:
            x = layer(x, edge_index)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x