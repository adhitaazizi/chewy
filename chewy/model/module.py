from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.norm import LayerNorm, MessageNorm

class GATModule(nn.Module):
    """
    DeeperGCN-style GAT layer with pre-activation residual connections and MsgNorm
    Following: Normalization → ReLU → GraphConv → Addition
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        heads: int = 1,
        dropout: float = 0.0,
        use_msg_norm: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_msg_norm = use_msg_norm
        
        # Pre-activation components
        self.norm = LayerNorm(in_channels)
        self.activation = nn.ReLU(inplace=True)
        
        # GAT layer - output should match input for residual connection
        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=False,
            residual=True
        )

        self.residual = nn.Identity()
            
    def forward(self, x, edge_index):
        # Store residual
        residual = self.residual(x)
        
        # Pre-activation: Normalization → ReLU → GraphConv
        x = self.norm(x)
        x = self.activation(x)
        gat_out = self.gat(x, edge_index)
        
        return residual + gat_out



