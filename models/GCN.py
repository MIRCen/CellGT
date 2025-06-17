import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, LayerNorm
from torch_geometric.nn import GCNConv




class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x,edge_index=data.x,data.edge_index
        x = F.dropout(x, p=0.02, training=self.training)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.02, training=self.training)
        x = self.conv2(x, edge_index)
        return x

