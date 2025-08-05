import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.norm import BatchNorm

class PhiLayer(nn.Module):
    def __init__(self, n_in, n_out, n_layer=2):
        super().__init__()
        self.conv = GINConv(nn.Identity(), train_eps=True)
        self.linears = nn.ModuleList([nn.Linear(n_in if i == 0 else n_out, n_out) for i in range(n_layer)])
        self.norms = nn.ModuleList([BatchNorm() for _ in range(n_layer)])
        self.activations = nn.ModuleList([nn.Relu(inplace=True) for _ in range(n_layer)])

    def reset_parameters(self):
        self.conv.reset_parameters()
        for linear, norm in zip(self.linears, self.norms): 
            linear.reset_parameters()
            norm.reset_parameters() 

    def forward(self, x):
        x = self.conv(x)
        for layer, norm, act in zip(self.linears, self.norms, self.activations):
            x = layer(x)
            x = norm(x)
            x = act(x)
        return x

class PhiModule(nn.Module):
    def __init__(self, n_in, n_out, n_layer=4):
        super().__init__()

        self.convs = nn.ModuleList([PhiLayer(n_in if i == 0 else n_out) for i in range(n_layer)])
        self.norms = nn.ModuleList([BatchNorm() for _ in range(n_layer)])
        self.activations = nn.ModuleList([nn.Relu(inplace=True) for _ in range(n_layer)])


    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms): 
            conv.reset_parameters()
            norm.reset_parameters() 

    def forward(self, x, edge_index):
        x = x.transpose(0, 1) 
        if mask is not None:
            mask = mask.transpose(0, 1) 
        previous_x = 0
        for conv, norm, act in zip(self.convs, self.norms, self.activations):
            x = conv(x, edge_index) # pass mask into
            x = norm(x)
            x = act(x)
            x += previous_x
            previous_x = x
        return x.transpose(0, 1)
    
