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
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        
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

    def __init__(self,io_channel,hidden_channel,kernel_size,bias=True):
        super(ResBlock, self).__init__()
        padding=kernel_size//2
        self.conv1=nn.Conv2d(io_channel,hidden_channel,kernel_size=kernel_size,stride=1,padding=padding,bias=bias)
        self.hidden=nn.Conv2d(hidden_channel,hidden_channel,kernel_size=kernel_size,stride=1,padding=padding,bias=bias)
        self.conv2=nn.Conv2d(hidden_channel,io_channel,kernel_size=kernel_size,stride=1,padding=padding,bias=bias)
        self.bn=nn.BatchNorm2d(io_channel)
        self.relu=nn.ReLU()
    def forward(self,x):
        x1=self.relu(self.conv1(x))
        x2=self.relu(self.hidden(x1))
        x3=self.relu(self.conv2(x2))
        x3=self.bn(x3)
        return x+x3


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
        self.preprocess=False
        
        # create CNN encoder
        self.encoder_layer=[]
        self.encoder_layer.append(nn.Sigmoid())
        for l in range(len(self.config['CNNEncoder']['kernel_size'])):
            inchannel=self.config['CNNEncoder']['channel'][l]
            outchannel=self.config['CNNEncoder']['channel'][l+1]
            kernel_size=self.config['CNNEncoder']['kernel_size'][l]
            padding=kernel_size//2
            self.encoder_layer.append(nn.Conv2d(inchannel,outchannel,kernel_size,stride=1,padding=padding))
            if self.config['CNNEncoder']['activation'] =='relu':
                activation=nn.ReLU()
            elif self.config['CNNEncoder']['activation'] =='leakyrelu':
                activation=nn.LeakyReLU()
            elif self.config['CNNEncoder']['activation'] =='tanh':
                activation=nn.Tanh()
            elif self.config['CNNEncoder']['activation'] =='sigmoid':
                activation=nn.Sigmoid()
            else:
                raise Exception('Unknown Activation Function')
            self.encoder_layer.append(activation)
        self.encoder_layer=nn.Sequential(*self.encoder_layer)
        self.flatten=nn.Flatten(start_dim=2)
        # create GAT
        self.gat=GNOLayers(self.config['Fusion'])

        self.decoder_layer=[]
        if "channel_att" in self.config.keys():
            self.channel_att=ChannelAttention(self.config['CNNDecoder']['input_channel'])
            self.decoder_layer.append(self.channel_att)
        
        
        if "RES" not in self.config['CNNDecoder'].keys():
            for l in range(len(self.config['CNNDecoder']['kernel_size'])):
                inchannel=self.config['CNNDecoder']['channel'][l]
                outchannel=self.config['CNNDecoder']['channel'][l+1]
                kernel_size=self.config['CNNDecoder']['kernel_size'][l]
                padding=kernel_size//2
                #padding value is -inf
                self.decoder_layer.append(nn.ConstantPad2d(padding,-999))
                self.decoder_layer.append(nn.Conv2d(inchannel,outchannel,kernel_size,stride=1,padding=0))
                if 'batch_norm' in self.config['CNNDecoder'].keys():
                    if self.config['CNNDecoder']['batch_norm']:
                        self.decoder_layer.append(nn.BatchNorm2d(outchannel))
                if self.config['CNNDecoder']['activation'] =='relu':
                    activation=nn.ReLU()
                elif self.config['CNNDecoder']['activation'] =='leakyrelu':
                    activation=nn.LeakyReLU()
                elif self.config['CNNDecoder']['activation'] =='tanh':
                    activation=nn.Tanh()
                elif self.config['CNNDecoder']['activation'] =='sigmoid':
                    activation=nn.Sigmoid()
                else:
                    raise Exception('Unknown Activation Function')
                self.decoder_layer.append(activation)
            if "dropout" in self.config['CNNDecoder'].keys():
                self.decoder_layer.append(nn.Dropout(self.config['CNNDecoder']['dropout']))
            self.decoder_layer.append(nn.Flatten(start_dim=1))
        else:
            if self.config['CNNDecoder']['RES']:
                for l in range(len(self.config['CNNDecoder']['kernel_size'])-1):
                    kernel_size=self.config['CNNDecoder']['kernel_size'][l]
                    inchannel=self.config['CNNDecoder']['input_channel']
                    hidden_channel=self.config['CNNDecoder']['hidden_channel'][l]
                    
                    #padding value is -inf
                    self.decoder_layer.append(ResBlock(inchannel,hidden_channel,kernel_size))
                
                kernel_size=self.config['CNNDecoder']['kernel_size'][-1]
                inchannel=self.config['CNNDecoder']['input_channel']
                outchannel=self.config['CNNDecoder']['output_channel']
                padding=kernel_size//2
                #padding value is -inf
                self.decoder_layer.append(nn.ConstantPad2d(padding,-999))
                self.decoder_layer.append(nn.Conv2d(inchannel,outchannel,kernel_size,stride=1,padding=0))
                if 'batch_norm' in self.config['CNNDecoder'].keys():
                    if self.config['CNNDecoder']['batch_norm']:
                        self.decoder_layer.append(nn.BatchNorm2d(outchannel))
                if "dropout" in self.config['CNNDecoder'].keys():
                    self.decoder_layer.append(nn.Dropout(self.config['CNNDecoder']['dropout']))
                self.decoder_layer.append(nn.Flatten(start_dim=1))

        # create MLP
        for l in range(len(self.config['MLP']['output_feature'])-1):
            in_dim=self.config['MLP']['output_feature'][l]
            out_dim=self.config['MLP']['output_feature'][l+1]

            self.decoder_layer.append(nn.Linear(in_dim,out_dim))
            if 'dropout' in self.config['MLP'].keys():
                self.decoder_layer.append(nn.Dropout(self.config['MLP']['dropout']))
            if 'batch_norm' in self.config['MLP'].keys():
                if self.config['MLP']['batch_norm']:
                    self.decoder_layer.append(nn.BatchNorm1d(out_dim))
            self.decoder_layer.append(nn.ReLU())
        if self.config['MLP']['output_feature'][-1]==5:
            self.decoder_layer[-1]=nn.Sigmoid()
        else:
            self.decoder_layer.pop()
        self.decoder_layer=nn.Sequential(*self.decoder_layer)
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
        
        
        if tb is not None:
            # visualize the input
            import matplotlib.pyplot as plt
            fig=plt.figure()
            plt.imshow(x[0,0,:,:].detach().cpu().numpy())
            plt.colorbar()
            plt.savefig('cnn_input_0.png')
            fig=plt.figure()
            plt.imshow(x[0,1,:,:].detach().cpu().numpy())
            plt.colorbar()
            plt.savefig('cnn_input_1.png')
            fig=plt.figure()
            plt.imshow(x[0,2,:,:].detach().cpu().numpy())
            plt.colorbar()
            plt.savefig('cnn_input_2.png')


        if self.config['CNNEncoder']['RES']:
            enc_out=self.encoder_layer(x)+x
        else:
            enc_out=self.encoder_layer(x)
        

        
        x=enc_out.view(B,N,enc_out.shape[1],enc_out.shape[2],enc_out.shape[3])
        
        # Flatten
        x=self.flatten(x)
        x=x.reshape(B,N,-1).permute(0,2,1)
        encoder_time=time.time()
        # GAT
        print(x.shape)
        x=self.gat(x)
        gat_time=time.time()
        
        # MLP
        if "CNNDecoder" in self.config.keys():
            x=x.reshape(B*N,-1,H,W)
            if test:
            # visualize the feature map
                import matplotlib.pyplot as plt
                fig=plt.figure()
                plt.imshow(x[0,0,:,:].detach().cpu().numpy())
                plt.legend()
                plt.savefig('cnn_feature_map_0.png')
                fig=plt.figure()
                plt.imshow(x[0,1,:,:].detach().cpu().numpy())
                plt.legend()
                plt.savefig('cnn_feature_map_1.png')
                fig=plt.figure()
                plt.imshow(x[0,2,:,:].detach().cpu().numpy())
                plt.legend()
                plt.savefig('cnn_feature_map_2.png')
            x=torch.cat([x,enc_out],dim=1)
        else:
            x=x.reshape(B*N,-1)

        x=self.decoder_layer(x)
        output=x.reshape(B,N,-1)
        
        return output
    
    