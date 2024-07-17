import utils.graphML as gml
import torch
import torch.nn as nn
import time

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=1):
        super(ChannelAttention, self).__init__()
        
        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveMaxPool2d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(in_channels, in_channels * reduction_ratio)
        self.fc2 = nn.Linear(in_channels * reduction_ratio, in_channels)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 全局平均池化
        avg_pool = self.global_avg_pool(x)
        
        # 压缩通道数 (B x C x 1 x 1)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        
        # 全连接层
        attention = self.fc1(avg_pool)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        
        # sigmoid激活函数
        attention = self.sigmoid(attention)
        
        # 扩展维度以便进行乘法操作
        attention = attention.unsqueeze(2).unsqueeze(3)
        
        # 重标定
        out = x * attention
        
        return out

class ResBlock(nn.Module):

    def __init__(self,in_channel,hidden_channel,out_channel,kernel_size,bias=True):
        super(ResBlock, self).__init__()
        padding=kernel_size//2
        self.res=(in_channel==out_channel)
        self.conv1=nn.Conv2d(in_channel,hidden_channel,kernel_size=kernel_size,stride=1,padding=padding,bias=bias)
        self.hidden=nn.Conv2d(hidden_channel,hidden_channel,kernel_size=kernel_size,stride=1,padding=padding,bias=bias)
        self.conv2=nn.Conv2d(hidden_channel,out_channel,kernel_size=kernel_size,stride=1,padding=padding,bias=bias)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU()
    def forward(self,x):
        x1=self.relu(self.conv1(x))
        x2=self.relu(self.hidden(x1))
        x3=self.relu(self.conv2(x2))
        x3=self.bn(x3)
        if self.res:
            return x+x3
        else:
            return x3


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
        self.attentionMode=config['attention_mode']
        
        

        

        downwardPass =[]
        
        
        for l in range(self.K):
            
            layer=gml.GraphFilterBatchAttentional(self.in_features[l]*self.heads[l], self.in_features[l+1], 2, self.heads[l+1], self.E, self.bias,attentionMode=self.attentionMode)
            
            downwardPass.append(layer)
            

        upwardPass = []
        
        # reversely iterate
        for l in range(self.K):

            layer=gml.GraphFilterBatchAttentional(self.out_features[l]*self.heads[-l-1], self.out_features[l+1], 2,self.heads[-l-1], self.E, self.bias,attentionMode=self.attentionMode)
            upwardPass.append(layer)

        shortcut = []
        for l in range(self.K):
            layer=gml.GraphFilterBatchAttentional(self.in_features[l]*self.heads[l], self.out_features[-l-1], 2, self.heads[l+1], self.E, self.bias,attentionMode=self.attentionMode)
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
        if self.config['CNNEncoder']['enable']:
            self.CNN_encoder=[]
            for l in range(len(self.config['CNNEncoder']['kernel_size'])):
                input_channel=self.config['CNNEncoder']['channel'][l]
                output_channel=self.config['CNNEncoder']['channel'][l+1]
                hidden_channel=self.config['CNNEncoder']['hidden_channel'][l]
                kernel_size=self.config['CNNEncoder']['kernel_size'][l]
                self.CNN_encoder.append(ResBlock(input_channel,hidden_channel,output_channel,kernel_size))
            if "dropout" in self.config['CNNEncoder'].keys():
                self.CNN_encoder.append(nn.Dropout(self.config['CNNEncoder']['dropout']))
            if self.config['CNNEncoder']['batch_norm']:
                self.CNN_encoder.append(nn.BatchNorm2d(self.config['CNNEncoder']['channel'][-1]))
            self.CNN_encoder.append(nn.AdaptiveMaxPool2d((1,1)))
            self.CNN_encoder=nn.Sequential(*self.CNN_encoder)

        if self.config['MLPEncoder']['enable']:
            self.MLP_encoder=[]
            for l in range(len(self.config['MLPEncoder']['output_feature'])-1):
                self.MLP_encoder.append(nn.Linear(self.config['MLPEncoder']['output_feature'][l],self.config['MLPEncoder']['output_feature'][l+1]))
                if 'dropout' in self.config['MLPEncoder'].keys():
                    self.MLP_encoder.append(nn.Dropout(self.config['MLPEncoder']['dropout']))
                self.MLP_encoder.append(nn.ReLU())
            if self.config['MLPEncoder']['batch_norm']:
                self.MLP_encoder.append(nn.BatchNorm1d(self.config['MLPEncoder']['output_feature'][-1]))
            self.MLP_encoder=nn.Sequential(*self.MLP_encoder)


        # create GAT
        self.gat=GNOLayers(self.config['Fusion'])

        # create CNN encoder
        if self.config['CNNDecoder']['enable']:
            self.CNN_decoder=[]
            if self.config['CNNDecoder']['channel_att']:
                self.CNN_decoder.append(ChannelAttention(self.config['CNNDecoder']['channel'][0]))

            
            for l in range(len(self.config['CNNDecoder']['kernel_size'])):
                input_channel=self.config['CNNDecoder']['channel'][l]
                output_channel=self.config['CNNDecoder']['channel'][l+1]
                hidden_channel=self.config['CNNDecoder']['hidden_channel'][l]
                kernel_size=self.config['CNNDecoder']['kernel_size'][l]
                self.CNN_decoder.append(ResBlock(input_channel,hidden_channel,output_channel,kernel_size))
            if "dropout" in self.config['CNNDecoder'].keys():
                self.CNN_decoder.append(nn.Dropout(self.config['CNNDecoder']['dropout']))
            if self.config['CNNDecoder']['batch_norm']:
                self.CNN_decoder.append(nn.BatchNorm2d(self.config['CNNDecoder']['channel'][-1]))
            self.CNN_decoder.append(nn.AdaptiveMaxPool2d((1,1)))
            self.CNN_decoder=nn.Sequential(*self.CNN_decoder)
        
        if self.config['MLPDecoder']['enable']:
            self.MLP_decoder=[]
            for l in range(len(self.config['MLPDecoder']['output_feature'])-1):
                self.MLP_decoder.append(nn.Linear(self.config['MLPDecoder']['output_feature'][l],self.config['MLPDecoder']['output_feature'][l+1]))
                if 'dropout' in self.config['MLPDecoder'].keys():
                    self.MLP_decoder.append(nn.Dropout(self.config['MLPDecoder']['dropout']))
                self.MLP_decoder.append(nn.ReLU())
            self.MLP_decoder.pop()
            if self.config['MLPDecoder']['batch_norm']:
                self.MLP_decoder.append(nn.BatchNorm1d(self.config['MLPDecoder']['output_feature'][-1]))
            self.MLP_decoder=nn.Sequential(*self.MLP_decoder)

    
    def init_params(self):

        #init using kaiming normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def add_graph(self, Slist):
        self.gat.addGSO(Slist)

    def forward(self, x, test=False,tb=None):
        # x: (B, N, C, H, W)
        # output: (B, N, Action)
        # B: batch size
        # N: number of agents
        # Action: number of actions

        # CNN Encoder
        start_time=time.time()
        B=x.shape[0]
        N=x.shape[1]
        C=x.shape[2]
        H=x.shape[3]
        W=x.shape[4]
        
        
        x=x.reshape(B*N,x.shape[2],x.shape[3],x.shape[4])
        
        if self.config['CNNEncoder']['enable']:
            x=self.CNN_encoder(x)
            x=x.view(B*N,-1)

        if self.config['MLPEncoder']['enable']:
            x=self.MLP_encoder(x)
        
        x=x.view(B,-1,N)


        fused_feature = self.gat(x)

        x=torch.cat([x,fused_feature],dim=1)
        x=x.view(B*N,-1)

        if self.config['CNNDecoder']['enable']:
            x=x.reshape(B*N,-1,H,W)
            x=self.CNN_decoder(x)
            x=x.reshape(B*N,-1)
        if self.config['MLPDecoder']['enable']:
            x=self.MLP_decoder(x)
        x=x.view(B,N,-1)
        
        return x
    
    