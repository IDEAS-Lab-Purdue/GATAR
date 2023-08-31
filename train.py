import torch
from dataset.dataset import myDataset
import argparse
import yaml
import os
from torch.utils.data import DataLoader
from model.GAT_ml import GATPlanner
from utils.setupEXP import start as st
import time
from agent.preprocessor.HetPreprocessor import Preprocessor
import random
import numpy as np

DIRECTIONS = torch.tensor([[0,0],[0, 1], [0, -1], [1, 0], [-1, 0]])
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str,default=None)
    parser.add_argument('--exp_dir',type=str,default=None)
    parser.add_argument('--con_train',action='store_true',default=False)
    parser.add_argument('--test',action='store_true',default=False)
    return parser.parse_args()
def weighted_loss(output,target):

    weight_mat=torch.tensor(
        [
        #0 stay 1 right 2 left 3 up 4 down
        [1,1,5,1,1],
        [1,1,1,1,1 ],  
        [1,5,1,1,1 ],
        [1,1,1,1,5 ],
        [1,1,1,5,1 ]
        ]
        )
    weight_mat=weight_mat.to(device)
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    losses = ce_loss(output,target)
    _, preds = torch.max(output, 1)
    for i in range(output.size(0)):
        losses[i] *= weight_mat[target[i], preds[i]]
    
    return losses.mean()
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
def all2supervised(task_pred,pos):
    B,N,H,W,_=task_pred.shape
    b_index=torch.arange(B)[:,None,None,None].long().to(device)
    N_index=torch.arange(N)[None,:,None,None].long().to(device)
    x_index=pos[:,:,0][:,:,None,None].long().to(device)
    y_index=pos[:,:,1][:,:,None,None].long().to(device)
    supervised_pred=task_pred[b_index,N_index,x_index,y_index].squeeze(2).squeeze(2)
    return supervised_pred

def train_epoch(epoch_num):
    model.train()
    epoch_loss=[]
    for i, data in enumerate(train_loader):
        obs,adj,pos,task = data
        adj=adj.to(device)
        obs=obs.to(device)
        task=task.to(device)
        pos=pos.to(device)
        

        # generate adjacency matrix from obs
        adj=adj.squeeze(1)
        SList=adj2SList(adj,config)
        model.add_graph(SList)
        
        #
        task_pred=model(obs)
        
        
        
        # B*N*H*W*2
        # use pos to get the task_pred
        supervised_pred=all2supervised(task_pred,pos)

        
        
        
        
        # obs: batch_size*num_agents*channel*H*W 
            # channel: 4 (by default)
            # 0: obstacle map --0/1
            # 1: target map --float gaussian distributions around observed targets
            # 2: cost map --float (currently not "extended" only a gaussian distribution centered at the ego agent)
            # 3: element-wise product of 1,2 --float

        
        
        loss = criterion(supervised_pred,task)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    

    epoch_loss=sum(epoch_loss)/len(epoch_loss)
    log.info(f'epoch {epoch_num} |train loss {epoch_loss}')

    return epoch_loss
def val_epoch():
    model.eval()
    epoch_loss=[]
    with torch.no_grad():
        total_num=0
        distances=[]
        for i, data in enumerate(val_loader):
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
def train():
    max_epoch = 5000
    best_loss=None
    
    for epoch in range(current_epoch,max_epoch):
        epoch_loss=train_epoch(epoch)
        tb_writer.add_scalar('train_loss',epoch_loss,epoch)
        scheduler.step()
        tb_writer.add_scalar('lr',scheduler.get_last_lr()[-1],epoch)
        save_dict={'epoch':epoch,'model_state_dict':model.state_dict()}
        torch.save(save_dict,os.path.join(exp_dir,'last_model.pth'))
        if epoch%10==0:
            epoch_loss,stat=val_epoch()
            print(f'epoch {epoch} loss {epoch_loss}')
            tb_writer.add_scalar('val_loss',epoch_loss,epoch)
            log.info(f'epoch {epoch} |VAL loss {epoch_loss}')
            log.info(f'epoch {epoch} |VAL 50% percentile distance {stat[0]}')
            log.info(f'epoch {epoch} |VAL 90% percentile distance {stat[1]}')
            log.info(f'epoch {epoch} |VAL mean distance {stat[2]}')
            tb_writer.add_scalar('val_50%_percentile_distance',stat[0],epoch)
            tb_writer.add_scalar('val_90%_percentile_distance',stat[1],epoch)
            
            if best_loss is None or epoch_loss<best_loss:
                best_loss=epoch_loss
                save_dict={'epoch':epoch,'model_state_dict':model.state_dict()}
                torch.save(save_dict,os.path.join(exp_dir,'best_model.pth'))
                log.info(f'best model saved at epoch {epoch}')

        
    
if __name__ == "__main__":
    args = parser()
    
    if args.con_train:
        exp_dir = args.exp_dir
        config_path = exp_dir+"/params.yaml"
        config = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)
        config.update({'data_root':args.data_root})
    else:
        config_path = args.data_root+"/params.yaml"
        config = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)
        config.update({'data_root':args.data_root})
        time_stamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        exp_dir = f'exp/{config["config_name"]}_agent_{config["env"]["num_agents"]}/{time_stamp}'
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        else:
            raise ValueError("experiment name already exists")
        # copy the config file to the experiment folder
        os.system(f'cp {config_path} {exp_dir}')
    log,tb_writer = st(exp_dir)

    train_dataset = myDataset(config,phase="train")
    val_dataset = myDataset(config,phase="val")
    log.info(f'train dataset size {len(train_dataset)}')
    log.info(f'val dataset size {len(val_dataset)}')


    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=12,prefetch_factor=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False,num_workers=12,prefetch_factor=16)

    model = GATPlanner(config)
    if "initialization" in config.keys() and not args.con_train:
        model.init_params()
    if args.con_train:
        save_dict = torch.load(os.path.join(args.exp_dir,'last_model.pth'))
        model.load_state_dict(save_dict['model_state_dict'])
        current_epoch = save_dict['epoch']
        log.info(f'continue training from epoch {current_epoch}')
    else:
        current_epoch = 0
    
    log.info('model:')
    log.info(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f'device: {device}')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],weight_decay=config['weight_decay'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    
    criterion = torch.nn.MSELoss()
    
    if config['loss']=="MSE":
        criterion = torch.nn.MSELoss()
    elif config['loss']=="weighted":
        criterion = weighted_loss
    else:
        raise NotImplementedError
    train()
    


    