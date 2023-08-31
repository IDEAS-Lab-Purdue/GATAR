import torch
import torch.nn as nn
from model.GAT_ml import GATPlanner
from utils.setupEXP import start as st
from model.action import actionhead
from dataset.dataset import myDataset
import argparse
import yaml
import os
import time

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str,default=None)
    parser.add_argument('--exp',type=str,default=None)
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

def train_epoch(epoch_num):
    
    action_model.train()
    epoch_loss=[]
    
    for i,batch_data in enumerate(train_loader):
        
        obs,adj,pos,task,action = batch_data
        adj=adj.to(device)
        obs=obs.to(device)
        task=task.to(device)
        pos=pos.to(device)
        action=action.to(device)
        
        adj=adj.squeeze(1)
        SList=adj2SList(adj,config)
        allocation_model.add_graph(SList)
        # task_pred=allocation_model(obs)
        # task_pred=task_pred.round().long()
        
        
        action_pred=action_model(obs,task)
        if action_pred.shape[-1]==2:
            pass
        elif action_pred.shape[-1]==5:
            action=co2index(action)
            
        loss=criterion(action_pred,action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    
    epoch_loss=sum(epoch_loss)/len(epoch_loss)
    log.info(f'epoch {epoch_num} train loss: {epoch_loss}')
    return epoch_loss

def val_epoch(epoch_num):
    action_model.eval()
    epoch_loss=[]
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            obs,adj,pos,task,action = batch_data
            
                
            adj=adj.to(device)
            obs=obs.to(device)
            task=task.to(device)
            pos=pos.to(device)
            action=action.to(device)
            adj=adj.squeeze(1)
            SList=adj2SList(adj,config)
            allocation_model.add_graph(SList)
            #task_pred=allocation_model(obs)
            action_pred=action_model(obs,task)
            if i==0:
                log.info(f'action pred: {action_pred[0]}')
                log.info(f'action: {action[0]}')
                #log.info(f'task pred: {task_pred[0]}')
                log.info(f'task: {task[0]}')
                log.info(f'pos: {pos[0]}')

            loss=criterion(action_pred,action)
            epoch_loss.append(loss.item())
    epoch_loss=sum(epoch_loss)/len(epoch_loss)
    log.info(f'epoch {epoch_num} | val loss: {epoch_loss}')
    return epoch_loss


if __name__ == '__main__':

    arg=parser()
    exp_root=arg.exp
    data_root=arg.data_root
    if exp_root is None:
        raise ValueError('exp_root is None')
    else:
        config = yaml.load(open(os.path.join(exp_root,'params.yaml'), 'r'), Loader=yaml.FullLoader)
    config.update({'data_root':data_root})

    dataset_Dir=os.path.join(data_root,'dataset_dict.pkl')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    train_dataset,val_dataset = myDataset(config, 'train',stage='action'),myDataset(config, 'val',stage='action')
    print("dataset loaded")
    print("train dataset size:",len(train_dataset))
    print("val dataset size:",len(val_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=12, prefetch_factor=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=12, prefetch_factor=8)

    allocation_model=GATPlanner(config)
    # load model
    save_dict=torch.load(os.path.join(exp_root,'best_model.pth'))
    allocation_model.load_state_dict(save_dict['model_state_dict'])
    # require grads False
    for param in allocation_model.parameters():
        param.requires_grad=False
    # set allocation model to eval mode
    allocation_model.eval()
    allocation_model.to(device)
    print("allocation model loaded, from best model at epoch:",save_dict['epoch'])

    time_stamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    exp_root = os.path.join(exp_root,f'action_{time_stamp}')
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)
    #copy the config file to the experiment folder
    os.system(f'cp {os.path.join(exp_root,"../params.yaml")} {exp_root}')

    log,tb_writer = st(exp_root)
    log.info("start training action head")
    log.info(f'allocation model:')
    log.info(allocation_model)
    log.info(f"allocation model loaded, from best model at epoch:{save_dict['epoch']}")
    # create action head
    action_model=actionhead(config)

    log.info(f'action model:')
    log.info(action_model)

    action_model.to(device)
    best_loss=None
    if config['network']['action_head']['MLP']['output_feature'][-1]==2:
        criterion=nn.MSELoss()
    elif config['network']['action_head']['MLP']['output_feature'][-1]==5:
        criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(action_model.parameters(),lr=1e-5)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.75)
    # training
    for epoch in range(5000):

        epoch_loss=train_epoch(epoch)
        tb_writer.add_scalar('action_train_loss',epoch_loss,epoch)
        scheduler.step()
        if epoch%10==0:
            val_loss=val_epoch(epoch)
            tb_writer.add_scalar('action_val_loss',val_loss,epoch)
            if best_loss is None or val_loss<best_loss:
                best_loss=val_loss
                save_dict={'epoch':epoch,'model_state_dict':action_model.state_dict()}
                torch.save(save_dict,os.path.join(exp_root,'best_action_model.pth'))
                log.info(f'best model saved at epoch {epoch}')
        save_dict={'epoch':epoch,'model_state_dict':action_model.state_dict()}
        torch.save(save_dict,os.path.join(exp_root,'last_action_model.pth'))
        




        
