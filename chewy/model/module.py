from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.norm import LayerNorm, msg_norm

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
        self.dropout = nn.Dropout(dropout)
        
        # GAT layer - output should match input for residual connection
        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels // heads,
            heads=heads,
            dropout=dropout,
            concat=True if out_channels != in_channels else True
        )
        
        # Message normalization
        if use_msg_norm:
            self.msg_norm = msg_norm(learn_scale=True)
        
        # Projection layer if dimensions don't match
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        # Store residual
        residual = x
        
        # Pre-activation: Normalization → ReLU → GraphConv
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Graph convolution
        gat_out = self.gat(x, edge_index)
        
        # Apply message normalization if enabled
        if self.use_msg_norm:
            gat_out = self.msg_norm(x, gat_out)
        
        # Handle dimension mismatch for residual connection
        if self.projection is not None:
            residual = self.projection(residual)
        
        # Residual connection: Addition
        return residual + gat_out



