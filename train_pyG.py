import torch
from dataset.datasetPyG import myDataset
import argparse
import yaml
import os
from torch.utils.data import DataLoader
from model.GAT import CustomGAT
from utils.setupEXP import start as st
import time

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str,default=None)
    parser.add_argument('--con_train',action='store_true',default=False)
    return parser.parse_args()

def train_epoch():
    model.train()
    epoch_loss=0
    for i, data in enumerate(train_loader):
        obs,pos,_,action = data
        pos=pos.to(device).float()
        obs=obs.to(device)
        action=action.to(device)
        
        action_pred = model(obs,pos).view(action.shape[0],5,-1)
        loss = criterion(action_pred,action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        acc=(action_pred.argmax(dim=1)==action).sum().item()/action.shape[0]
    return epoch_loss*batch_size,acc
def val_epoch():
    model.eval()
    epoch_loss=0
    with torch.no_grad():
        acc_num=0
        total_num=0
        for i, data in enumerate(val_loader):
            obs,pos,_,action = data
            pos=pos.to(device).float()
            obs=obs.to(device)
            action=action.to(device)
            
            action_pred = model(obs,pos).view(action.shape[0],5,-1)
            loss = criterion(action_pred,action)
            epoch_loss+=loss.item()
            acc_num+=(action_pred.argmax(dim=1)==action).sum().item()
            total_num+=action.shape[0]*5
        acc=acc_num/total_num
    return epoch_loss*batch_size,acc

def train():
    max_epoch = 5000
    best_acc=0
    for epoch in range(max_epoch):
        epoch_loss,acc=train_epoch()
        tb_writer.add_scalar('train_loss',epoch_loss,epoch)
        scheduler.step()
        tb_writer.add_scalar('lr',scheduler.get_lr()[0],epoch)
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
        
if __name__ == "__main__":
    args = parser()
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
    
    train_dataset = myDataset(config,phase="train")
    val_dataset = myDataset(config,phase="val")
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    print(f'train dataset size {len(train_dataset)}')
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)
    print(f'val dataset size {len(val_dataset)}')

    
    model = CustomGAT(config)
    log,tb_writer = st(exp_dir)
    log.info('model:')
    log.info(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss()
    train()
    


    