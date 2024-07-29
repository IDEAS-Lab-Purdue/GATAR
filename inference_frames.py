import torch
from dataset.dataset import myDataset
import argparse
import yaml
import os
from torch.utils.data import DataLoader
from model.GAT_ml import GATPlanner
from utils.setupEXP import start as st
import time
from env.grids import gridWorld
import json
import random
import numpy as np
import imageio
from agent.team import *

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',type=str,default=None)
    parser.add_argument('--data',type=str,default=None)
    return parser.parse_args()

def adj2SList(adj,config):
    SList=[adj]
    for l in range(config['network']['Fusion']['K']-1):
        adj = torch.bmm(adj,adj)
        SList.append(adj)
    if "NormS" in config['network'].keys():
            if config['network']['NormS']:
                for i in range(len(SList)):
                    S=SList[i]
                    eigenvalues=torch.linalg.eig(S)[0]
                    eigenvalues=torch.real(eigenvalues)
                    max_eigenvalue=torch.max(eigenvalues,dim=1)[0].unsqueeze(1).unsqueeze(1)
                    S=S/max_eigenvalue
                    SList[i]=S
        
    if "non_ego" in config['network'].keys():
            if config['network']['non_ego']:
                for i in range(len(SList)):
                    S=SList[i]
                    # set diagonal elements to 0
                    S[:,range(S.shape[1]),range(S.shape[1])]=0
                    SList[i]=S
    SList=torch.stack(SList,dim=1)
    return SList
def val_epoch(data_loader):
    model.eval()
    epoch_loss=[]
    with torch.no_grad():
        total_num=0
        distances=[]
        for i, data in enumerate(data_loader):
            obs,adj,pos,task = data
            adj=adj.to(device)
            obs=obs.to(device)
            task=task.to(device)
            pos=pos.to(device)
            
            adj=adj.squeeze(1)
            SList=adj2SList(adj,config)
            model.add_graph(SList)
            task_pred=model(obs) #B*N*2
            task_pred=all2supervised(task_pred,pos)
            # calculate the distance between prediction and ground truth
            distance=torch.norm(task_pred-task,dim=2) #B*N
            distance=distance.view(-1) #B*N
            distances.append(distance)


            output_num=random.randint(0,task_pred.shape[0]-1)
            
            loss = criterion(task_pred,task)
            epoch_loss.append(loss.item())
    
    distances=torch.cat(distances,dim=0)
    distances=distances.cpu().numpy()
    # calculate the 50% percentile distance, average distance, and 90% percentile distance
    percentile50=np.percentile(distances,50)
    percentile90=np.percentile(distances,90)
    mean_distance=np.mean(distances)
    stat=[percentile50,percentile90,mean_distance]


    epoch_loss=sum(epoch_loss)/len(epoch_loss)
    return epoch_loss,stat

if __name__ == "__main__":
    args = parser()
    
    exp_dir = args.exp_dir
    config_path = exp_dir+"/params.yaml"
    config = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)
    config.update({'data_root':args.data_root})
    
    exp_dir = exp_dir+"/inference"
    log,tb_writer = st(exp_dir)

    test_dataset = myDataset(config, 'test')
    log.info(f'test dataset size: {len(test_dataset)}')


    batch_size = 4
    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model = GATPlanner(config)
    
    save_dict = torch.load(os.path.join(args.exp_dir,'last_model.pth'))
    model.load_state_dict(save_dict['model_state_dict'])
    current_epoch = save_dict['epoch']
    log.info(f'testing model from epoch {current_epoch}')

    log.info('model:')
    log.info(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f'device: {device}')
    model.to(device)
    val_epoch(data_loader)

    