import numpy as np
import tcod
from tcod import libtcodpy

def create_agents(params):
    agents=[]
    for name,spec in params.items():
        print("initiating",name)
        
        for i in range(spec['num']):
            agents.append(agent(spec))
    return agents

class agent:

    def __init__(self,params):
        
        self.params = params
        self.type = params['type']
        self.pos = np.zeros((2,1))
        self.sensing_range = params['sensing_range']
        if self.type == 'UAV':
            self.__create_UAV()
        elif self.type == 'UGV':
            self.__create_UGV()
        else:
            raise NotImplementedError
    
    def __create_UAV(self):
        self.observe = self.UAVobserve

    def __create_UGV(self):
        self.observe = self.UGVobserve

    def UAVobserve(self,grid):

        temp_grid=grid.copy()
        temp_grid[:,:,0]=np.zeros_like(grid[:,:,0])
        trans=np.where(temp_grid[:,:,0]==1,False,True)
        my_pos=self.pos
        mask=tcod.map.compute_fov(trans,[my_pos[0],my_pos[1]],radius=self.sensing_range)
        repl_mask=np.repeat(mask[:,:,np.newaxis],3,axis=2)
        output= np.full(grid.shape,999)
        output[repl_mask]=grid[repl_mask]
        return output

        
    def UGVobserve(self,grid):
        
        my_pos=self.pos
        trans=np.where(grid[:,:,0]==1,False,True)
        mask=tcod.map.compute_fov(trans,[my_pos[0],my_pos[1]],radius=self.sensing_range)
        repl_mask=np.repeat(mask[:,:,np.newaxis],3,axis=2)
        output= np.full(grid.shape,999)
        output[repl_mask]=grid[repl_mask]
        return output

        