from typing import List

import torch
import torch.nn as nn
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from chewy.model.module import GATModule
    
class ChewyEncoder(nn.Module):
    """
    DeeperGCN-style GAT Encoder with deep architecture capabilities
    """
    def __init__(
        self,
        node_feat_name: str,
        edge_index_name: str,
        input_dim: int,
        dim_list: List[int],
        heads: int = 1,
        dropout: float = 0.0,
        use_msg_norm: bool = True,
        pe_k: int = 8,  # Number of Laplacian eigenvectors for positional encoding
        use_pe: bool = True,  # Whether to use positional encoding
    ):
        super().__init__()
        
        self.node_feat_name = node_feat_name
        self.edge_index_name = edge_index_name
        self.input_dim = input_dim
        self.dim_list = dim_list
        self.use_pe = use_pe
        
        # Add Laplacian Eigenvector Positional Encoding
        if self.use_pe:
            self.pe_transform = AddLaplacianEigenvectorPE(
                k=pe_k,
                attr_name='laplacian_eigenvector_pe',
                is_undirected=True  # Assuming undirected graphs for efficiency
            )
            # Adjust input dimension to account for PE features
            effective_input_dim = input_dim + pe_k
        else:
            self.pe_transform = None
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
                use_msg_norm=use_msg_norm
            )
        )
        
        # Intermediate layers
        for i in range(len(dim_list) - 1):
            self.layers.append(
                GATModule(
                    in_channels=dim_list[i],
                    out_channels=dim_list[i + 1],
                    heads=heads,
                    dropout=dropout,
                    use_msg_norm=use_msg_norm
                )
            )
        
        # Final normalization (as suggested in DeeperGCN)
        self.final_norm = nn.LayerNorm(dim_list[-1])
    
    def forward(self, data):
        # Apply Laplacian Eigenvector PE if enabled
        if self.use_pe and self.pe_transform is not None:
            data = self.pe_transform(data)
            # Concatenate original features with PE features
            if hasattr(data, 'laplacian_eigenvector_pe'):
                x = torch.cat([data.x, data.laplacian_eigenvector_pe], dim=-1)
            else:
                x = data.x
        else:
            x = data.x
        
        edge_index = data.edge_index
        
        # Pass through all layers
        for layer in self.layers:
            x = layer(x, edge_index)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x