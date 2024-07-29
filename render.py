import env.grids
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
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,default='configs/params.yaml')
    parser.add_argument('--render',action='store_true',default=False)
    return parser.parse_args()

args = parser()

configs = yaml.load(open(args.data,'r'),Loader=yaml.FullLoader)
print(configs)
name=configs['config_name']
size_x=configs['env']['grid_size']['x']
size_y=configs['env']['grid_size']['y']
agent_num = configs['env']['num_agents']    
path =f'data/{name}_{size_x}x{size_y}_agent{agent_num}/'
#check if experiment name exists
if not os.path.exists(path+'map_dict.json') or not os.path.exists(path+'params.yaml'):
    os.system(f'python3 utils/create_env.py --config {args.data}')
# wait for the environment to be created
while not os.path.exists(path+'map_dict.json'):
    pass
# load the map
map_dict = json.load(open(path+'map_dict.json','r'))
agent_num = configs['env']['num_agents']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
configs.update({'device':device})
# create team
team = MRS(configs)
# create environment using stored data
instance = env.grids.gridWorld(configs['env'],map_dict['0'])

print("initialization done")

# exit

# render the process to a video

model = GATPlanner(configs)
model=model.to(device)
print("model created")
print(model)
start_time = time.time()
frames = []
for round in tqdm(range(1)):
    for step in tqdm(range(400)):

        instance.step(team,model)
        if args.render:
            frame = instance.vis(team.agents_observation)
            frames.append(frame)

    instance.reset()
finish_time = time.time()
print("average time per step: (unit: s)",(finish_time-start_time)/400)
print("average time per round: ",(finish_time-start_time)/1)





if args.render:
    print("start rendering")
    imageio.mimsave(f'{path}render.mp4', frames, duration=0.1)
    print("rendering done")