import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv as GATConv
import torch.nn as nn
import yaml

class CustomGAT(torch.nn.Module):
    def __init__(self, config):
        super(CustomGAT, self).__init__()
        self.config = config
        self.HW=config['network']['x']*config['network']['y']
        # 1. Node feature encoding with CNN
        CNN_config = config["network"]["CNNEncoder"]
        if CNN_config["activation"] == "relu":
            activation = nn.ReLU()
        elif CNN_config["activation"] == "leakyrelu":
            activation = nn.LeakyReLU()
        elif CNN_config["activation"] == "tanh":
            activation = nn.Tanh()
        elif CNN_config["activation"] == "sigmoid":
            activation = nn.Sigmoid()
        else:
            raise Exception("Unknown Activation Function")
        CNN_layers = []
        for i in range(len(CNN_config["kernel_size"])):
            l=nn.Conv2d(CNN_config["channel"][i], CNN_config["channel"][i+1], kernel_size=CNN_config["kernel_size"][i], stride=1, padding=CNN_config["kernel_size"][i]//2)
            CNN_layers.append(l)
            CNN_layers.append(activation)
        self.cnn_encoder = nn.Sequential(*CNN_layers)
        self.CNN_config = CNN_config
        
        # 2. Multi-layer GAT network
        Fusion_config = config["network"]["Fusion"]
        assert Fusion_config["K"]==len(Fusion_config["channel"])-1

        self.gat_convs = torch.nn.ModuleList()
        self.gat_convs.append(GATConv(Fusion_config['channel'][0]*self.HW, Fusion_config['channel'][1]*self.HW, heads=Fusion_config['heads']))
        for i in range(Fusion_config["K"]-1):
            self.gat_convs.append(GATConv(Fusion_config['heads']*self.HW * Fusion_config["channel"][i+1], 
                                          Fusion_config["channel"][i+2]*self.HW, heads=Fusion_config['heads']))
        self.Fusion_config = Fusion_config
        # 3. Decoding node features to 5D action vector

        decoder_config = config["network"]["MLP"]   
        self.cnn_decoder = nn.Sequential(
            nn.Conv2d(Fusion_config['heads']*Fusion_config['channel'][-1], decoder_config['channel'], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            # ... (Add more layers if needed)
        )
        self.mlp_decoder = nn.Sequential(
            nn.Linear(config["network"]["x"] * config["network"]["y"] * decoder_config["channel"], decoder_config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        self.decoder_config = decoder_config

        self.commRange = config["env"]["comm_range"]

        # 4. Initialize weights
        if "init_weights" in config["network"].keys():
            self.init_weights()
        # 5. finish init
        print("Finish Init")
    
    def init_weights(self):

        # use kaiming normal initialization for CNN
        for m in self.cnn_encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias.data, 0)
        for m in self.cnn_decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias.data, 0)
        # use xavier normal initialization for MLP
        for m in self.mlp_decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                nn.init.constant_(m.bias.data, 0)
        


    def forward(self, x, pos):
        # x: [Batch, Node, H, W, C]
        # pos: [Node, 2]

        # 1. Encode node features
        batch_size, num_nodes, H, W, C = x.size()
        x = x.view(-1, C, H, W)  # Reshape to [Batch*Node, C, H, W]
        x = self.cnn_encoder(x)  # Output: [Batch*Node, CHANNEL_CNN, H, W]

        # 2. Construct edge_index from pos using torch.cdist
        distances = torch.cdist(pos, pos)  # Shape: [Batch, Node, Node]
        mask = (distances > 0) & (distances <= self.commRange)

        batches, rows, cols = mask.nonzero(as_tuple=True)
        # Adjust node indices for batched data
        rows += batches * num_nodes
        cols += batches * num_nodes
        edge_index = torch.stack([rows, cols], dim=0)
        x=x.view(batch_size*num_nodes,-1)

        # 3. Apply GAT layers
        for gat_conv in self.gat_convs:
            x = gat_conv(x, edge_index)
            x = F.elu(x)

        x= x.view(-1, self.Fusion_config['channel'][-1]*self.Fusion_config['heads'], self.config['network']['x'], self.config['network']['y'])
        
        # 4. Decode node features to action vectors
        x = self.cnn_decoder(x)
        x = x.view(batch_size, num_nodes, -1)
        x = self.mlp_decoder(x)

        return x