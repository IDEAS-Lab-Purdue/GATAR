import h5py
import argparse
import numpy as np
import os
import torch

def load_data_h5(data_root):
    h5_path = os.path.join(data_root, 'dataset.h5')
    with h5py.File(h5_path, 'r') as f:
        obs = f['obs'][:]
        agent_pos = f['agent_pos'][:]
        allocated_tasks = f['allocated_tasks'][:]
        actions = f['actions'][:]
    return obs, agent_pos, allocated_tasks, actions

if __name__ == '__main__':
    
    merging_data = ['dataset/test_15x15_agent2',
                    'dataset/test_15x15_agent4',
                    'dataset/test_15x15_agent6',
                    'dataset/test_15x15_agent8',
                    'dataset/test_15x15_agent10']
                    