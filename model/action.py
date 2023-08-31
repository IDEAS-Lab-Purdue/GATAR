import torch
import torch.nn as nn

class actionhead(nn.Module):

    def __init__(self, config):
        super(actionhead, self).__init__()
        self.config = config['network']['action_head']

        cnn_layers = []
        cnn_layers.append(nn.Conv2d(self.config['CNN']['input_channel'], self.config['CNN']['hidden_channel'][0], kernel_size=self.config['CNN']['kernel_size'][0], stride=1, padding=self.config['CNN']['kernel_size'][0]//2))
        cnn_layers.append(nn.ReLU())
        for l in range(0,len(self.config['CNN']['kernel_size'])-2):
            cnn_layers.append(nn.Conv2d(self.config['CNN']['hidden_channel'][l], self.config['CNN']['hidden_channel'][l+1], kernel_size=self.config['CNN']['kernel_size'][l+1], stride=1, padding=self.config['CNN']['kernel_size'][l+1]//2))
            cnn_layers.append(nn.ReLU())
        cnn_layers.append(nn.Conv2d(self.config['CNN']['hidden_channel'][-1], self.config['CNN']['output_channel'], kernel_size=self.config['CNN']['kernel_size'][-1], stride=1, padding=self.config['CNN']['kernel_size'][-1]//2))
        cnn_layers.append(nn.ReLU())
        self.cnn = nn.Sequential(*cnn_layers)

        mlp_layers = []
        for l in range(len(self.config['MLP']['output_feature'])-1):
            mlp_layers.append(nn.Linear(self.config['MLP']['output_feature'][l], self.config['MLP']['output_feature'][l+1]))
            mlp_layers.append(nn.ReLU())
        mlp_layers.pop()
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self,obs,tgt):
        obs=self.processing_tgt_batch(obs,tgt)
        obs = obs.view(obs.shape[0]*obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
        output=self.cnn(obs).view(obs.shape[0],-1)
        output=self.mlp(output)
        output=output.view(tgt.shape[0],tgt.shape[1],-1)

        return output

    def processing_tgt_batch(self,obs,tgt):
        cost_map = obs[:,:,2:3,:,:]
        tgt_pos_map = torch.zeros_like(cost_map).to(obs.device)
        tgt[tgt<0]=0
        tgt[tgt>=obs.shape[3]]=obs.shape[3]-1
        
        B=obs.shape[0]
        N=obs.shape[1]

        B_indices = torch.arange(B)[:, None, None, None].long()
        N_indices = torch.arange(N)[None, :, None, None].long()
        tgt=tgt.long()
        y_indices = tgt[:, :, 0][:, :, None, None]
        x_indices = tgt[:, :, 1][:, :, None, None]  

        tgt_pos_map[B_indices, N_indices,0, x_indices,y_indices] = 1
        obstacle_map = obs[:,:,0:1,:,:]

        new_obs=torch.cat([obstacle_map,cost_map,tgt_pos_map],dim=2)
        return new_obs
