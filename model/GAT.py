# implement a graph attention network using pytorch geometric, used for generating node features

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import networkx as nx
import matplotlib.pyplot as plt

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout, alpha):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, out_dim, dropout=dropout)

    def forward(self, x, edge_index):
        print(x)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x


