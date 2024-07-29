import torch
import os
import pickle
from numba import jit
import numpy as np
import h5py

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

def load_data(data_root):
    h5_path = os.path.join(data_root, 'dataset.h5')
    with h5py.File(h5_path, 'r') as f:
        obs = f['obs'][:]
        agent_pos = f['agent_pos'][:]
        allocated_tasks = f['allocated_tasks'][:]
        actions = f['actions'][:]
    return obs, agent_pos, allocated_tasks, actions

class myDataset(torch.utils.data.Dataset):
    def __init__(self, config, phase):
        self.config = config
        
        obs, agent_pos, allocated_tasks, actions = load_data(config['data_root'])
        
        if phase == 'train':
            self.obs = obs[:int(len(obs)*0.8)]
            self.agent_pos = agent_pos[:int(len(agent_pos)*0.8)]
            self.allocated_tasks = allocated_tasks[:int(len(allocated_tasks)*0.8)]
            
        else:
            self.obs = obs[int(len(obs)*0.8):]
            self.agent_pos = agent_pos[int(len(agent_pos)*0.8):]
            self.allocated_tasks = allocated_tasks[int(len(allocated_tasks)*0.8):]
        
        

    def __len__(self):
        return len(self.obs)
    


    def __getitem__(self, index):
        obs = self.obs[index]
        obs = torch.tensor(obs, dtype=torch.float32).permute(0,3,1,2)
        # N,2
        agent_pos = self.agent_pos[index]
        agent_pos = torch.tensor(agent_pos, dtype=torch.float32)
        
        # N,N
        adj = generate_adjacency_matrix(self.config['env']['comm_range'], agent_pos)
        adj = torch.tensor(adj, dtype=torch.float32)
        # N,2
        allocated_tasks = self.allocated_tasks[index]
        
        
        return obs,adj,agent_pos,allocated_tasks