import env.grids
import yaml
import numpy as np
from tqdm import tqdm
import imageio


# read configs from yaml file

configs=yaml.load(open('config/test.yaml','r'),Loader=yaml.FullLoader)

agent_num = configs['env']['num_agents']
# create environment
instance = env.grids.gridWorld(configs['env'])

# render the process to a video
frames=[]
for round in tqdm(range(1)):
    for step in tqdm(range(200)):

        action=np.random.randint(0,5,agent_num)
        instance.step(action)
        frame=instance.vis(draw_arrows=True)
        frames.append(frame)
    
    instance.reset()

#print(instance.get_local_observation(0))

# save the video
imageio.mimsave('test.gif', frames, 'GIF', duration = 0.1)


