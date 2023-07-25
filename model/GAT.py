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

# testing the code
if __name__ == '__main__':

    model=GAT(1, 10, 1, 10, 0.5, 0.2)
    
    #initialize the node features
    x = torch.randn(10, 1)
    #initialize the edge index
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 5, 6, 7, 8],
                                 [1, 0, 2, 1, 4, 3, 6, 5, 8, 7]], dtype=torch.long)

    # visualize the graph with nodes and edges, using networkx and matplotlib
    # showing node features
    
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                        (5, 6), (6, 7), (7, 8), (8, 0)])
    # label it with node features
    node_labels = {0: x[0][0], 1: x[1][0], 2: x[2][0], 3: x[3][0], 4: x[4][0],
                    5: x[5][0], 6: x[6][0], 7: x[7][0], 8: x[8][0]}
    nx.set_node_attributes(G, node_labels, 'node_features')
    # visualize the graph
    nx.draw(G, with_labels=True, font_weight='bold',labels=node_labels)
    # save figure
    plt.savefig('graph.png')
    plt.clf()
    out = model(x, edge_index)

    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                        (5, 6), (6, 7), (7, 8), (8, 0)])
    # label it with node features
    node_labels = {0: x[0][0], 1: x[1][0], 2: x[2][0], 3: x[3][0], 4: x[4][0],
                    5: x[5][0], 6: x[6][0], 7: x[7][0], 8: x[8][0]}
    print(node_labels)
    nx.set_node_attributes(G, node_labels, 'node_features')
    # visualize the graph with node_features
    nx.draw(G, with_labels=True, font_weight='bold',labels=node_labels)
    # save figure
    plt.savefig('graph_out.png')

