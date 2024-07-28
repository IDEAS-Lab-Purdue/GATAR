import h5py
import argparse
import numpy as np
import os
import torch
import pickle
from numba import jit


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

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    #parser.add_argument('--comm_range', type=float, default=1.0)
    return parser

def main(args):
    
    
    
    data_path = os.path.join(args.data_root,"dataset_dict.pkl")
    
    h5_path = os.path.join(args.data_root, 'dataset.h5')
    if os.path.exists(h5_path):
        print(f"Dataset already exists at {h5_path}")
        return
    # load data from pickle file one by one
    data = []
    obs = []
    agent_pos = []
    allocated_tasks = []
    actions = []
    with open(data_path, 'rb') as f:
        while True:
            try:
                data_point = pickle.load(f)
                obs.append(data_point['obs'])
                agent_pos.append(data_point['agent_pos'])
                allocated_tasks.append(data_point['allocated_tasks'])
                actions.append(data_point['actions'])
                print(f"Loading data: {len(obs)}", end='\r')
            except EOFError:
                break
    # combine list of list into one list
    # data = [item for sublist in data for item in sublist]
    
    print(f"Dataset size: {len(obs)}, successfully loaded")
    keys = ['obs', 'agent_pos', 'allocated_tasks', 'actions']
    np_obs = np.array(obs)
    np_agent_pos = np.array(agent_pos)
    np_allocated_tasks = np.array(allocated_tasks)
    np_actions = np.array(actions)

    with h5py.File(h5_path, 'w') as f:
        for i, key in enumerate(keys):
            f.create_dataset(key, data=eval(f'np_{key}'))
    print(f"Saved data to {h5_path}")

def load_data(data_root):
    h5_path = os.path.join(data_root, 'dataset.h5')
    with h5py.File(h5_path, 'r') as f:
        obs = f['obs'][:]
        agent_pos = f['agent_pos'][:]
        allocated_tasks = f['allocated_tasks'][:]
        actions = f['actions'][:]
    return obs, agent_pos, allocated_tasks, actions

if __name__ == '__main__':
    args = arg_parser().parse_args()
    main(args=args)
    
    import time
    start_time = time.time()
    obs, agent_pos, allocated_tasks, actions = load_data(args.data_root)
    end_time = time.time()
    print(obs.shape, agent_pos.shape, allocated_tasks.shape, actions.shape)
    print(f"used time: {end_time-start_time}")