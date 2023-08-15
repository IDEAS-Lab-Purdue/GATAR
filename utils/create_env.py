import numpy as np
import matplotlib.pyplot as plt
import random
import yaml
import os
import json
import argparse

def assign_obstacles():
        # assign obstacles to each subgroup
        obstacles = []
        subgroup_x_index = np.linspace(0,params['grid_size']['x'],params['sub_groups']['side_division']+1,dtype=int)
        subgroup_y_index = np.linspace(0,params['grid_size']['y'],params['sub_groups']['side_division']+1,dtype=int)
        
        for i in range(params['sub_groups']['side_division']):
            for j in range(params['sub_groups']['side_division']):
                if 'group_obstacle_matrix' not in params['sub_groups']:
                    num_obstacles = random.randint(0,params['obstacles']['max_num'])
                else:
                    num_obstacles = params['sub_groups']['group_obstacle_matrix'][i][j]
                for k in range(num_obstacles):
                    obstacles.append([random.randint(subgroup_x_index[i],subgroup_x_index[i+1]-1),
                                        random.randint(subgroup_y_index[j],subgroup_y_index[j+1]-1)])

        return obstacles

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default='config/test.yaml')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parser()
    params = yaml.load(open(args.config,'r'),Loader=yaml.FullLoader)
    map_num = 1000
     
    size_x = params['env']['grid_size']['x']
    size_y = params['env']['grid_size']['y']
    name=params['config_name']
    agent_num = params['env']['num_agents']
     
    data_path = f'data/{name}_{size_x}x{size_y}_agent{agent_num}/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # copy the params.yaml to data folder
    os.system(f'cp {args.config} {data_path}params.yaml')

    map_dict = {}
    params= params['env']
    for i in range(map_num):
        obstacles = assign_obstacles()
        map_dict.update({i:obstacles})
    
    # save the map_dict to json file
    with open(f'{data_path}map_dict.json','w') as f:
        json.dump(map_dict,f)


    


     
    