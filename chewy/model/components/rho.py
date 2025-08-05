import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.norm import LayerNorm

class RhoModule:
    def __init__(self, n_in, n_out, num_layers, heads=4, dropout=0.0):
        super().__init__()

        self.convs = nn.ModuleList([GATv2Conv(
            in_channels=n_in,
            out_channels=n_out,
            heads=heads,
            dropout=dropout
        ) for i in range(num_layers)])
        self.norms = nn.ModuleList([LayerNorm() for _ in range(num_layers)])
        self.activations = nn.ModuleList([nn.Relu(inplace=True)])

    def forward(self, x, edge_index):
        for conv, norm, act in zip(self.convs, self.norms, self.activations):
            x = conv(x, edge_index)
            x = norm(x)
            x = act(x)
        return x