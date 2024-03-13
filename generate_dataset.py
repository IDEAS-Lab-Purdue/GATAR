from env.grids import gridWorld
import yaml
import numpy as np
from tqdm import tqdm
import imageio
import json
import argparse
import os
from agent.team import *
import time
import threading
import queue
import multiprocessing as mp
import tensorboard
import logging
from utils.setupEXP import *
from math import sqrt
import pickle

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default='config/test.yaml')
    parser.add_argument('--render',action='store_true',default=False)
    parser.add_argument('--load_pretrained',type=str,default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()

    configs = yaml.load(open(args.config,'r'),Loader=yaml.FullLoader)
    print("configs loaded from ",args.config)
    configs['network']['load_pretrained']=args.load_pretrained
    
    name=configs['config_name']
    size_x=configs['env']['grid_size']['x']
    size_y=configs['env']['grid_size']['y']
    agent_num = configs['env']['num_agents']    
    dir =f'dataset/{name}_{size_x}x{size_y}_agent{agent_num}/'
    
    #check if experiment name exists
    if not os.path.exists(dir+'map_dict.json') or not os.path.exists(dir+'params.yaml'):
        print(f'creating new experiment {name} with {agent_num} agents')
        os.system(f'python3 utils/create_env.py --config {args.config} --map_num 200')
    else:
        print(f'loading experiment {name} with {agent_num} agents')
    # wait for the environment to be created
    while not os.path.exists(dir+'map_dict.json'):
        pass

    if os.path.exists(dir+'dataset_dict.pkl'):
        print("dataset already exists")
        exit()
    if os.path.exists(dir+'prioritized_dataset_dict.pkl'):
        print("prioritized dataset already exists")
        exit()

    # load the map
    map_dict = json.load(open(dir+'map_dict.json','r'))
    agent_num = configs['env']['num_agents']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log,tb_writer = start(dir)
    log.info(f'generating datasets for {name} with {agent_num} agents')
    log.info(f'using device {device}')
    configs.update({'device':device})
    dataset_dict={}
    prioritized_dataset_dict={}
    try:
        store_history = configs['preprocess']['history']
    except:
        store_history = False
    
    for key in tqdm(map_dict.keys()):
        
        for k in range(100):
            data_points = []
            prioritized_data_points = []
            single_map = map_dict[key]
            log.info(f'generating data for map {key}')
            env = gridWorld(configs['env'],single_map)
            team = MRS(configs)
            team.reset()
            cost = np.ones([env.grid.shape[0],env.grid.shape[1]]).astype(int)
            cost_air = np.ones([env.grid.shape[0],env.grid.shape[1]]).astype(int)
            cost[env.grid[:,:,0]==1]=0
            if "diag" in configs['env'].keys():
                if configs['env']['diag']:
                    diag = 1
                else:
                    diag = 0
            else:
                diag = 0
            tcod_graph = tcod.path.SimpleGraph(cost=cost, cardinal=1, diagonal = diag)
            tcod_graph_air = tcod.path.SimpleGraph(cost=cost_air, cardinal=1, diagonal = diag)
            env.sync(team)
            env.step(team,np.zeros([team.agent_num]).astype(int))
            signal = True
            allocated_tasks = np.zeros([team.agent_num,2])
            found_path = [None]*team.agent_num
            frames = []
            history_position = []

            
            while env.targets.shape[0]!=0:

                if signal: # allocate tasks
                    targets = env.targets
                    agents_pos = team.agents_pos
                    dist_array = torch.zeros([agents_pos.shape[0],targets.shape[0]])
                    
                    for i in range(agents_pos.shape[0]):
                        agent_pos = agents_pos[i,:]

                        agent_type=team.agents_type[i]
                        if agent_type=='UAV':
                            temp_graph=tcod_graph_air
                        elif agent_type=='UGV':
                            temp_graph=tcod_graph
                        else:
                            raise Exception('Unknown Agent Type')

                        pf = tcod.path.Pathfinder(temp_graph)
                        pf.add_root((agent_pos[0],agent_pos[1]))
                        for j in range(targets.shape[0]):
                            target = targets[j,:]
                            path = pf.path_to((target[0],target[1]))
                            dist_array[i,j] = len(path)
                    
                    processed_agent=[]
                    for _ in range(agents_pos.shape[0]):

                        linear_index = torch.argmin(dist_array)
                        row, col = linear_index // dist_array.shape[1], linear_index % dist_array.shape[1]
                        allocated_tasks[row,:] = targets[col,:]
                        # delete this column and row
                        
                        
                        dist_array[:,col] = torch.tensor([999]*dist_array.shape[0])
                        processed_agent.append(row)
                        dist_array[processed_agent,:] = torch.tensor([float('inf')]*dist_array.shape[1])
                        agent_type=team.agents_type[row]
                        if agent_type=='UAV':
                            temp_graph=tcod_graph_air
                        elif agent_type=='UGV':
                            temp_graph=tcod_graph
                        else:
                            raise Exception('Unknown Agent Type')
                        
                        pf = tcod.path.Pathfinder(temp_graph)
                        pf.add_root((agents_pos[row,0],agents_pos[row,1]))
                        found_path[row] = pf.path_to((targets[col,0],targets[col,1]))
                        #print("assigned agent",row,"to target",col,"with distance",len(found_path[row]))
                        

                    signal = False
                
                actions=[]
                

                for i in range(team.agent_num):
                    
                    if found_path[i].shape[0]==1:
                        actions.append(0)
                        continue
                    next_pos = found_path[i][1]
                    action = next_pos - team.agents_pos[i,:]
                    # delete the first element in the path
                    if action[0]==0 and action[1]==0:
                        action = 0
                    elif action[0]==0 and action[1]==1:
                        action = 1
                    elif action[0]==0 and action[1]==-1:
                        action = 2
                    elif action[0]==1 and action[1]==0:
                        action = 3
                    elif action[0]==-1 and action[1]==0:
                        action = 4
                    elif action[0]==1 and action[1]==1:
                        action = 5
                    elif action[0]==1 and action[1]==-1:
                        action = 6
                    elif action[0]==-1 and action[1]==1:
                        action = 7
                    elif action[0]==-1 and action[1]==-1:
                        action = 8
                    else:
                        raise Exception('Unknown Action')
                    actions.append(action)
                    found_path[i]=np.delete(found_path[i],0,axis=0)
                    if found_path[i].shape[0]==1:
                        signal = True
                
                actions = np.array(actions)
                obs=team.observe(env.grid)
                agent_pos = team.agents_pos

                # store datapoint
                # adapt tensor to int tensor
                
                data_point={'obs':obs.numpy().astype(int),
                            'agent_pos':agent_pos.astype(int),
                            'grid':env.grid.copy().astype(int),
                            'actions':actions.astype(int)}
                if store_history:
                    history_position.append(agent_pos)
                    data_point.update({'history_position':history_position.copy()})
                data_points.append(data_point)
                if args.render and key=='0' and k==0:
                    frame=env.vis(obs)
                    frames.append(frame)
                # store prioritized datapoint
                if signal and len(data_points)>5:
                    prioritized_data_points.append(data_points[-3])
                    prioritized_data_points.append(data_points[-2])
                    prioritized_data_points.append(data_points[-1])
                # step the environment
                env.step(team,actions)
            # save the datapoints 
            #print("cost",len(data_points),"for map",key)
            
            token=key+str(k)+str(time.time())
            token=hash(token)
            
            # dump the data points to file
            with open(dir+'dataset_dict.pkl', 'ab') as f:
                pickle.dump(data_points, f)

            #dataset_dict.update({token:data_points})

            # dump the prioritized data points to file
            with open(dir+'prioritized_dataset_dict.pkl', 'ab') as f:
                pickle.dump(prioritized_data_points, f)
            #prioritized_dataset_dict.update({token: prioritized_data_points})
            if args.render and key=='0' and k==0:
                imageio.mimsave(dir+'map0.gif', frames, 'GIF', duration=1)

    # save the dataset_dict to json file
    # torch.save(dataset_dict,dir+'dataset_dict.pt')
    # torch.save(prioritized_dataset_dict,dir+'prioritized_dataset_dict.pt')
    log.info(f'dataset saved to {dir}')