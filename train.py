import torch
from dataset.dataset import myDataset
import argparse
import yaml
import os
from torch.utils.data import DataLoader
from model.GAT_ml import GATPlanner
from utils.setupEXP import start as st
import time

import numpy as np

DIRECTIONS = torch.tensor([[0,0],[0, 1], [0, -1], [1, 0], [-1, 0]])
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str,default=None)
    parser.add_argument('--exp_dir',type=str,default=None)
    parser.add_argument('--con_train',action='store_true',default=False)
    parser.add_argument('--test',action='store_true',default=False)
    return parser.parse_args()
def train_epoch():
    model.train()
    epoch_loss=[]
    for i, data in enumerate(train_loader):
        obs,adj,grid,action = data
        adj=adj.to(device)
        obs=obs.to(device)
        action=action.to(device)
        # generate adjacency matrix from obs
        adj=adj.squeeze(1)
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
        model.add_graph(SList)
        
        
        
        if "loss" in config.keys():
            if config['loss']=="MSE":
                action_pred = model(obs)
                DIRECTIONS = torch.tensor([[0,0],[0, 1], [0, -1], [1, 0], [-1, 0]]).float().to(device)
                action = DIRECTIONS[action]
            elif config['loss']=="weighted":
                action_pred = model(obs).view(action.shape[0],5,-1)
                
            else:
                raise NotImplementedError
        else:
            action_pred = model(obs).view(action.shape[0],config['network']['MLP']['output_feature'][-1],-1)
        
        
        loss = criterion(action_pred,action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        if "loss" in config.keys():
            acc=0
        else:
            acc=(action_pred.argmax(dim=1)==action).sum().item()/action.shape[0]
    
    epoch_loss=sum(epoch_loss)/len(epoch_loss)
    log.info(f'train loss {epoch_loss}')

    return epoch_loss,acc
def val_epoch():
    model.eval()
    epoch_loss=[]
    with torch.no_grad():
        acc_num=0
        total_num=0
        for i, data in enumerate(val_loader):
            obs,adj,grid,action = data
            adj=adj.to(device)
            obs=obs.to(device)
            action=action.to(device)
            # generate adjacency matrix from obs
            adj=adj.squeeze(1)
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
                        S[:,range(S.shape[1]),range(S.shape[1])]=0
                        SList[i]=S
            
            SList=torch.stack(SList,dim=1)
            model.add_graph(SList)
            
            
            
            if "loss" in config.keys():
                if config['loss']=="MSE":
                    action_pred = model(obs)
                    DIRECTIONS = torch.tensor([[0,0],[0, 1], [0, -1], [1, 0], [-1, 0]]).to(device)
                    
                    action = DIRECTIONS[action]
                elif config['loss']=="weighted":
                    action_pred = model(obs,args.test).reshape(action.shape[0],5,-1)
                    
                    
                else:
                    raise NotImplementedError
            else:
                action_pred = model(obs,args.test).reshape(action.shape[0],config['network']['MLP']['output_feature'][-1],-1)
            
            loss = criterion(action_pred,action)
            # log.info("action_pred_prob: {}".format(action_pred[0]))
            # log.info("action_pred: {}".format(action_pred[0].argmax(dim=0)))
            # log.info("action_gt: {}".format(action[0]))
            epoch_loss.append(loss.item())
            if "loss" in config.keys():
                if config['loss']=="MSE":
                    acc=0
                else:
                    acc=(action_pred.argmax(dim=1)==action).sum().item()
            else:
                acc=(action_pred.argmax(dim=1)==action).sum().item()
            acc_num+=acc
            total_num+=action.shape[0]*action.shape[1]
            if args.test:
                break
        acc=acc_num/total_num
        epoch_loss=sum(epoch_loss)/len(epoch_loss)
    return epoch_loss,acc

def train():
    max_epoch = 5000
    best_acc=0
    if args.test:
        val_epoch()
        max_epoch=current_epoch
    for epoch in range(current_epoch,max_epoch):
        epoch_loss,acc=train_epoch()
        tb_writer.add_scalar('train_loss',epoch_loss,epoch)
        scheduler.step()
        tb_writer.add_scalar('lr',scheduler.get_lr()[0],epoch)
        save_dict={'epoch':epoch,'model_state_dict':model.state_dict()}
        torch.save(save_dict,os.path.join(exp_dir,'last_model.pth'))
        if epoch%10==0:
            epoch_loss,acc=val_epoch()
            print(f'epoch {epoch} loss {epoch_loss}')
            tb_writer.add_scalar('val_loss',epoch_loss,epoch)
            tb_writer.add_scalar('val_acc',acc,epoch)
            log.info(f'epoch {epoch} loss {epoch_loss} acc {acc}')
            if acc>best_acc:
                best_acc=acc
                save_dict={'epoch':epoch,'model_state_dict':model.state_dict()}
                torch.save(save_dict,os.path.join(exp_dir,'best_model.pth'))
                log.info(f'best model saved at epoch {epoch}')

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=16,prefetch_factor=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False,num_workers=16,prefetch_factor=8)
    # visulize the data
    for i, data in enumerate(train_loader):
        obs,adj,grid,action = data
        obs_sample=obs[0]
        action_sample=action[0]
        from matplotlib import pyplot as plt
        fig=plt.figure()
        for j in range(3):
            ax=fig.add_subplot(1,3,j+1)
            ax.set_xticks(np.arange(0,16,1))
            ax.set_yticks(np.arange(0,16,1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True)
            ax.set_xlim(0,15)
            ax.set_ylim(0,15)
            ax.set_aspect('equal')
            observation_test=obs_sample[j]
            local_obstacles = np.where(observation_test[0,:,:]==1)
            for i in range(len(local_obstacles[0])):
                ax.add_patch(plt.Rectangle((local_obstacles[0][i],local_obstacles[1][i]),1,1,fill=True,color='k'))
            # draw observed targets map
            plt.imshow(observation_test[2,-1:,:],cmap='Reds',alpha=0.5)
            # add a dim yellow background on the observed area
            observed_area = np.where(observation_test[0,:,:]<0)
            for i in range(len(observed_area[0])):
                ax.add_patch(plt.Rectangle((observed_area[0][i],observed_area[1][i]),1,1,fill=True,color='black',alpha=0.1))
            # draw the ego agent
            ego_pos=np.where(observation_test[1,:,:]==1)
            ax.plot(ego_pos[0][0]+0.5,ego_pos[1][0]+0.5,'b^',markersize=1)
            ax.set_title(f'action {action_sample[j]}')
        
        fig.savefig(os.path.join(exp_dir,'obs.png'))
        break

    

    model = GATPlanner(config)
    if "initialization" in config.keys() and not args.con_train:
        model.init_params()
    if args.con_train:
        save_dict = torch.load(os.path.join(args.exp_dir,'best_model.pth'))
        model.load_state_dict(save_dict['model_state_dict'])
        current_epoch = save_dict['epoch']
        log.info(f'continue training from epoch {current_epoch}')
    else:
        current_epoch = 0
    log.info('model:')
    log.info(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],weight_decay=config['weight_decay'])
    if "lr_reset" in config.keys():
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=3e-6)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    
    
    criterion = torch.nn.CrossEntropyLoss()
    if "loss" in config.keys():
        if config['loss']=="MSE":
            criterion = torch.nn.MSELoss()
        elif config['loss']=="weighted":
            criterion = weighted_loss
        else:
            raise NotImplementedError
    train()
    


    