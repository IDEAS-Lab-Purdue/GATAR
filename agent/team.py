import torch
from model.GAT_ml import GATPlanner
import multiprocessing as mp
import tcod
from tcod import libtcodpy
import threading
import numpy as np

class MRS():

    def __init__(self,config):
        self.config = config
        self.agents_type = []
        self.agents_pos = None
        self.sensing_ranges = []
        self.agents_observation = None
        self.create_agents(config['agent'])
        if self.agents_pos is None or len(self.agents_pos)!=self.agents_pos.shape[0]:
            raise Exception('Agents Not Properly Created')
        self.agents_observation =np.zeros([len(self.agents_pos),config['env']['grid_size']['x'],config['env']['grid_size']['y'],3])
        print('MRS Created')
        print(self.agents_pos.shape)
        print(self.agents_observation.shape)
        print(self.agents_type)

    def create_agents(self,params):

        total_num = 0
        for spec in self.config['agent'].values():
            numInspec=spec['num']
            for i in range(numInspec):
                self.agents_type.append(spec['type'])
                self.sensing_ranges.append(spec['sensing_range'])
                total_num+=1
        self.agents_pos = np.zeros([total_num,2]).astype(int)
        
    def __single_observe(self,grid,agent_id):
        #grid: H*W*C
        #agent_id: i
        #return: 1*H*W*C
        agent_pos=self.agents_pos[agent_id]
        agent_type=self.agents_type[agent_id]
        if agent_type=='UAV':
            obs=self.__UAVObserve(grid,agent_pos,self.sensing_ranges[agent_id])
        elif agent_type=='UGV':
            obs=self.__UGVObserve(grid,agent_pos,self.sensing_ranges[agent_id])
        else:
            raise Exception('Unknown Agent Type')
        
        self.agents_observation[agent_id]=obs
        

    def __UAVObserve(self,grid,agent_pos,sensing_range):
        #grid: H*W*C
        #agent_pos: 1*2
        #return: 1*H*W*C
        temp_grid=grid.copy()
        temp_grid[:,:,0]=np.zeros_like(grid[:,:,0])
        trans=np.where(temp_grid[:,:,0]==1,False,True)
        
        mask=tcod.map.compute_fov(trans,[agent_pos[0],agent_pos[1]],radius=sensing_range)
        repl_mask=np.repeat(mask[:,:,np.newaxis],3,axis=2)
        output= np.full(grid.shape,999)
        output[repl_mask]=grid[repl_mask]
        return output

    def __UGVObserve(self,grid,agent_pos,sensing_range):
        #grid: H*W*C
        #agent_pos: 1*2
        #return: 1*H*W*C
        trans=np.where(grid[:,:,0]==1,False,True)
        mask=tcod.map.compute_fov(trans,[agent_pos[0],agent_pos[1]],radius=sensing_range)
        repl_mask=np.repeat(mask[:,:,np.newaxis],3,axis=2)
        output= np.full(grid.shape,999)
        output[repl_mask]=grid[repl_mask]
        return output

    def observe(self,grid):

        #grid: H*W*C
        #return: N*H*W*C
        # get the observation of all agents parallelly 
        for i in range(len(self.agents_type)):

            self.__single_observe(grid,i)

        return torch.FloatTensor(self.agents_observation)

        

    
    def step(self,grid,model):
        
        obs=self.observe(grid) #N*H*W*C
        obs=obs.permute(0,3,1,2) #N*C*H*W
        obs=obs.unsqueeze(0) #1*N*C*H*W
        
        adj_mat=torch.eye(len(self.agents_type)).unsqueeze(0).unsqueeze(0) #B*1*N*N
        SList=[adj_mat]*self.config['network']['Fusion']['K']
        model.add_graph(SList)
        obs=obs.to(self.config['device'])
        action=model(obs)
        action=action.squeeze(0).detach().cpu().numpy()
        action=np.argmax(action,axis=1)
        
        
        return action
        
        

    