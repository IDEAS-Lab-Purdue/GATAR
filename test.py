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

if __name__ == '__main__':
    args = parser()

    exp_root = args.exp
    data_root = args.data

    config = yaml.load(open(os.path.join(exp_root,'params.yaml'), 'r'), Loader=yaml.FullLoader)
    config['agent']['UAV1']['num']=1
    map_path = os.path.join(data_root,'map_dict.json')
    model = GATPlanner(config)
    # load model
    save_dict=torch.load(os.path.join(exp_root,'best_model.pth'))
    
    model.load_state_dict(save_dict['model_state_dict'])
    print("model loaded from epoch {}".format(save_dict['epoch']))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model.to(device)    
    # load map from json
    map_dict = json.load(open(map_path,'r'))
    print("map loaded from {}".format(map_path))
    # random select a map
    map_id = random.choice(list(map_dict.keys()))
    print("map id: {}".format(map_id))
    map = map_dict[map_id]
    # generate environment
    env = gridWorld(config['env'],map)
    print("env generated from map {}".format(map_id))
     # initialize agents
    team = MRS(config)
    team.reset()
    print("agents reset")

    env.sync(team)
    print("agents initialized")
    # print info
    print("targets: {}".format(env.targets))
    print("obstacles: {}".format(env.obstacles))
    print("agents_pos: {}".format(team.agents_pos))
    frames=[]
    with torch.no_grad():
        for t in range(100):
            # visualize
            frame=env.vis(team.agents_observation)
            frames.append(frame)
            # observe
            obs=team.observe(env.grid).unsqueeze(0).permute(0,1,4,2,3).to(device)
            if 'preprocess' in config.keys():
                if config['preprocess']:
                    from agent.preprocessor.HetPreprocessor import Preprocessor
                    preprocessor=Preprocessor()
                    B,N,C,H,W=obs.shape
                    obs=obs.view(B*N,C,H,W)
                    obs=preprocessor(obs).view(B,N,C,H,W)
            # get adj matrix
            adj_mat=team.get_adj_mat()
            # get adj_list
            adj=adj_mat.unsqueeze(0)
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
            print("SList shape: {}".format(SList.shape))
            model.add_graph(SList)
            
            print("obs shape: {}".format(obs.shape))
            action = model(obs)

            print("action: {}".format(action))
            # action shape:
            print("action shape: {}".format(action.shape))

            prob_action=[]

            #random select based on probability
            for agent in range(action.shape[1]):
                prob_action.append(torch.multinomial(action[0,agent],1).unsqueeze(0).cpu())
                
            
            action=np.array(prob_action)
            
            action=action.astype(int)
            print("action: {}".format(action))  
            _,done=env.step(team,action)
            print("targets: {}".format(env.targets.tolist()))

            #input("press enter to continue")

            # check if done
            if done:
                print("done")
                break

    imageio.mimsave(os.path.join(exp_root,'{}.gif'.format(map_id)),frames,duration=0.5)