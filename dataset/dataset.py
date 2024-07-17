import torch
import os
import pickle
from numba import jit
import numpy as np

@jit(nopython=True,cache=True)
def generate_adjacency_matrix(comm_range, agent_pos):
    
    agent_pos = agent_pos.astype(np.float32)
    adj = np.zeros((agent_pos.shape[0], agent_pos.shape[0]))
    for i in range(agent_pos.shape[0]):
        for j in range(agent_pos.shape[0]):
            if i==j:
                adj[i,j] = 1
            dist = np.linalg.norm(agent_pos[i]-agent_pos[j])
            if dist<=comm_range:
                adj[i,j] = 1
    return adj
    
class myDataset(torch.utils.data.Dataset):
    def __init__(self, config, phase):
        self.config = config
        data_path = os.path.join(config['data_root'],"dataset_dict.pkl")
        # load data from pickle file one by one
        self.data = []
        with open(data_path, 'rb') as f:
            while True:
                # if len(self.data)>100:
                #     break
                try:
                    self.data.append(pickle.load(f))
                    
                except EOFError:
                    break
        # combine list of list into one list
        self.data = [item for sublist in self.data for item in sublist]
        
        if phase == 'train':
            self.data = self.data[:int(len(self.data)*0.8)]
        else:
            self.data = self.data[int(len(self.data)*0.8):]
        
        

    def __len__(self):
        return len(self.data)
    


    def __getitem__(self, index):
        x = self.data[index]
        # N H W 3
        obs = x['obs']
        obs = torch.tensor(obs, dtype=torch.float32).permute(0,3,1,2)
        # N,2
        agent_pos = x['agent_pos']
        # N,N
        adj = generate_adjacency_matrix(self.config['env']['comm_range'], agent_pos)
        adj = torch.tensor(adj, dtype=torch.float32)
        # H,W,3
        grid = x['grid']
        grid = torch.tensor(grid, dtype=torch.float32)
        # N
        actions = x['actions']
        
        
        return obs,adj,grid,actions