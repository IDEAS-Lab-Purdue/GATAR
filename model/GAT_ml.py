import utils.graphML as gml
import torch
import torch.nn as nn
import time

class GNOLayers(nn.Module):

    def __init__(self, config, bias=False):

        super(GNOLayers, self).__init__()
        self.in_features = config['input_feature']
        self.out_features = config['output_feature']
        if self.in_features[-1] != self.out_features[0]:
            raise ValueError("Input and output feature dimensions don't match")
        self.K = config['K']
        self.heads=[1]
        for l in range(self.K):
            self.heads.append(config['heads'])
        self.E=1
        self.bias=True
        self.SList=None

        

        downwardPass =[]
        
        
        for l in range(self.K):
            
            layer=gml.GraphFilterBatchAttentional(self.in_features[l]*self.heads[l], self.in_features[l+1], self.K, self.heads[l+1], self.E, self.bias,attentionMode="KeyQuery")
            
            downwardPass.append(layer)
            

        upwardPass = []
        
        # reversely iterate
        for l in range(self.K):

            layer=gml.GraphFilterBatchAttentional(self.out_features[l]*self.heads[-l-1], self.out_features[l+1], self.K,self.heads[-l-1], self.E, self.bias,attentionMode="KeyQuery")
            upwardPass.append(layer)

        shortcut = []
        for l in range(self.K):
            layer=gml.GraphFilterBatchAttentional(self.in_features[l]*self.heads[l], self.out_features[-l-1], self.K, self.heads[l+1], self.E, self.bias,attentionMode="KeyQuery")
            shortcut.append(layer)
        
        self.downwardPass = nn.ModuleList(downwardPass)
        self.upwardPass = nn.ModuleList(upwardPass)
        self.shortcut = nn.ModuleList(shortcut)

    def addGSO(self, Slist):
        for l in range(self.K):
            self.downwardPass[l].addGSO(Slist[:,l:l+1,:])  # add GSO for GraphFilter
            self.upwardPass[l].addGSO(Slist[:,self.K-l-1:self.K-l,:])  # add GSO for GraphFilter
            self.shortcut[l].addGSO(Slist[:,l:l+1,:])  # add GSO for GraphFilter
        
    
    def forward(self,extractFeatureMap):
        # forwarding through GNO
        propagatingFeature = [extractFeatureMap]
        for l in range(self.K):
            # downward pass
            propagatingFeature.append(self.downwardPass[l](propagatingFeature[-1]))
            
        
        for l in range(self.K):
            
            propagatingFeature.append(self.upwardPass[l](propagatingFeature[-1])+self.shortcut[-l-1](propagatingFeature[self.K-1-l]))
            

        #print(len(propagatingFeature))
        sharedFeature = propagatingFeature[-1]
        return sharedFeature




class GATPlanner(nn.Module):

    def __init__(self, config):
        super(GATPlanner, self).__init__()
        self.config = config['network']
        self.H=self.config['x']
        self.W=self.config['y']
        # create CNN encoder
        self.encoder_layer=[]
        for l in range(len(self.config['CNNEncoder']['kernel_size'])):
            inchannel=self.config['CNNEncoder']['channel'][l]
            outchannel=self.config['CNNEncoder']['channel'][l+1]
            kernel_size=self.config['CNNEncoder']['kernel_size'][l]
            padding=kernel_size//2
            self.encoder_layer.append(nn.Conv2d(inchannel,outchannel,kernel_size,stride=1,padding=padding))
            self.encoder_layer.append(nn.ReLU())
        self.encoder_layer=nn.Sequential(*self.encoder_layer)
        self.flatten=nn.Sequential(nn.Flatten(),nn.Linear(self.H*self.W*self.config['CNNEncoder']['channel'][-1],self.config['Flatten']['output_feature']),nn.ReLU())
        # create GAT
        self.gat=GNOLayers(self.config['Fusion'])

        self.decoder_layer=[]
        # create MLP
        for l in range(len(self.config['MLP']['output_feature'])-1):
            in_dim=self.config['MLP']['output_feature'][l]
            out_dim=self.config['MLP']['output_feature'][l+1]

            self.decoder_layer.append(nn.Linear(in_dim,out_dim))

            self.decoder_layer.append(nn.ReLU())
        self.decoder_layer[-1]=nn.Softmax()
        self.decoder_layer=nn.Sequential(*self.decoder_layer)

    def add_graph(self, Slist):
        self.gat.addGSO(Slist)

    def forward(self, x):
        # x: (B, N, C, H, W)
        # output: (B, N, Action)
        # B: batch size
        # N: number of agents
        # Action: number of actions

        # CNN Encoder
        
        start_time=time.time()
        B=x.shape[0]
        N=x.shape[1]
        x=x.reshape(B*N,x.shape[2],x.shape[3],x.shape[4])
        if self.config['CNNEncoder']['RES']:
            x=self.encoder_layer(x)+x
        else:
            x=self.encoder_layer(x)
        # Flatten
        x=self.flatten(x)
        x=x.reshape(B,-1,N)
        encoder_time=time.time()
        # GAT
        
        x=self.gat(x)
        gat_time=time.time()
        # MLP
        x=x.reshape(B*N,-1)
        x=self.decoder_layer(x)
        output=x.reshape(B,N,-1)
        
        return output
    
    