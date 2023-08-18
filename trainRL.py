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
from agent.DQN import *
import tensorboard
import logging
from utils.setupEXP import *

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,default='configs/params.yaml')
    parser.add_argument('--render',action='store_true',default=False)
    parser.add_argument('--load_pretrained',type=str,default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()

    configs = yaml.load(open(args.data,'r'),Loader=yaml.FullLoader)
    print("configs loaded from ",args.data)
    configs['network']['load_pretrained']=args.load_pretrained
    
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
    log,tb_writer = start(path)
    log.info(f'using device {device}')
    configs.update({'device':device})

    #create the DQN agents（networks）

    DQNAgent = DQNAgent(configs)

    #create MRS agents

    mrs= MRS(configs)

    best_reward=0
    #create the environment
    for episode in range(configs['RL']['episodes']):
        if episode>100 and DQNAgent.epsilon>configs['RL']['epsilon_min']:
            DQNAgent.epsilon*=configs['RL']['epsilon_decay']
        #create the environment
        map_num = random.sample(map_dict.keys(),1)[0]
        random_map=map_dict[map_num]
        print(f'episode {episode} loaded map {map_num}')
        
        instance = env.grids.gridWorld(configs['env'],random_map)
        #reset env and mrs
        instance.reset()
        mrs.reset()
        instance.agent_pos = mrs.agents_pos

        #get the initial state
        obs, SList = get_state(instance,mrs)
        episode_loss=0
        for step in range(configs['RL']['max_steps']):
            

            #get the action
            action = DQNAgent.select_action(obs,SList,mrs)

            #take the action
            reward,done = instance.step(mrs,action)
            #get the next state
            next_obs,next_SList = get_state(instance,mrs)
            #get the reward

            #store the experience
            DQNAgent.store_transition(obs,SList,action,reward,next_obs,next_SList,done)
            #update the network
            step_loss=DQNAgent.learn()
            episode_loss+=step_loss
            
            #update the state
            obs=next_obs
            SList=next_SList
            #check if the episode is done
            if done:
                break
        
        print(f'episode {episode+1} finished in {step+1} steps, episode loss {episode_loss}')
        log.info(f'episode {episode+1} finished in {step+1} steps, episode loss {episode_loss}')
        tb_writer.add_scalar('loss',episode_loss,episode+1)
        # get lr
        tb_writer.add_scalar('lr',DQNAgent.optimizer.param_groups[0]['lr'],episode+1)
        # get epsilon
        tb_writer.add_scalar('epsilon',DQNAgent.epsilon,episode+1)
        if (episode+1) % 10 == 0:
            instance.reset()
            mrs.reset()
            instance.agent_pos = mrs.agents_pos
            #testing the agent
            print('Testing!')
            map_num = random.sample(map_dict.keys(),1)[0]
            random_map=map_dict[map_num]
            print(f'testing on map {map_num}')
            instance = env.grids.gridWorld(configs['env'],random_map)
            obs, SList = get_state(instance,mrs)
            frames=[]
            rewards=0
            with torch.no_grad():
                for step in range(configs['RL']['max_steps']):
                    #visualize the environment
                    if args.render and (episode+1) % 100 == 0:
                        frame=instance.vis(mrs.agents_observation)
                        frames.append(frame)

                    action = DQNAgent.select_action(obs,SList,mrs,0)
                    
                    reward,done = instance.step(mrs,action)
                    rewards+=reward
                    obs, SList = get_state(instance,mrs)
                    if done:
                        break
            print(f'testing finished in {step+1} steps, total reward {rewards}')
            log.info(f'testing finished in {step+1} steps, total reward {rewards}')
            tb_writer.add_scalar('test_reward',rewards,episode+1)
            
            if rewards>best_reward:
                best_reward=rewards
                torch.save(DQNAgent.target_network.state_dict(),path+'best_model.pth')
            
            if (episode+1) % 50 == 0:
                torch.save(DQNAgent.target_network.state_dict(),path+f'model_{episode+1}.pth')

            if args.render and (episode+1) % 100 == 0:
                imageio.mimsave(f'data/{name}_{size_x}x{size_y}_agent{agent_num}/test_{episode+1}.gif',frames,duration=0.5)