import env.grids
import yaml
import numpy as np
from tqdm import tqdm
import imageio
import json
import argparse
from agent.agent import *

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,default='data/12x12/')
    return parser.parse_args()

args = parser()
configs = yaml.load(open(f'{args.data}params.yaml','r'),Loader=yaml.FullLoader)
map_dict = json.load(open(f'{args.data}map_dict.json','r'))

agent_num = configs['env']['num_agents']
# create agents instance
agents_list = create_agents(configs['agent'])
# create environment using stored data
instance = env.grids.gridWorld(configs['env'],map_dict['0'],agents_list)

# render the process to a video
frames=[]
for round in tqdm(range(1)):
    for step in tqdm(range(400)):

        action=np.random.randint(0,5,agent_num)
        instance.step(action)
        frame=instance.vis(draw_arrows=False)
        frames.append(frame)
        # save seperate frame
        if step%10==0:
            imageio.imwrite(f'test{step}.png',frame)
    
    instance.reset()

#print(instance.get_local_observation(0))

# save the video
imageio.mimsave('test.gif', frames, 'GIF', duration = 0.1)


