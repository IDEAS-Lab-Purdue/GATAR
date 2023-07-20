
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# base class for environment

class Env():

    def __init__(self):
        pass
        
    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

# extended class for grid world
class gridWorld(Env):

    def __init__(self,params):
        super().__init__()
        self.params = params
        self.grid = np.zeros((params['grid_size']['x'],params['grid_size']['y'],3))
        # divided into subgroups
        self.num_subgroups = params['sub_groups']['side_division']**2
        self.subgroup_x_index = np.linspace(0,params['grid_size']['x'],params['sub_groups']['side_division']+1,dtype=int)
        self.subgroup_y_index = np.linspace(0,params['grid_size']['y'],params['sub_groups']['side_division']+1,dtype=int)
        
        #visualize the grid
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks(np.arange(0,params['grid_size']['x']+1,1))
        self.ax.set_yticks(np.arange(0,params['grid_size']['y']+1,1))
        self.ax.grid(True)
        self.ax.set_xlim(0,params['grid_size']['x'])
        self.ax.set_ylim(0,params['grid_size']['y'])
        # add different color for different subgroups
        for i in range(params['sub_groups']['side_division']):
            for j in range(params['sub_groups']['side_division']):
                self.ax.add_patch(plt.Rectangle((self.subgroup_x_index[i],self.subgroup_y_index[j]),
                                                self.subgroup_x_index[i+1]-self.subgroup_x_index[i],self.subgroup_y_index[j+1]-self.subgroup_y_index[j],
                                                fill=True,color=np.random.rand(3,1).flatten(),alpha=0.5))
                # make the patch a square
                self.ax.set_aspect('equal')
                
                
        plt.show()
        # save the figure
        self.fig.savefig('grid.png')
        

        

        # randomly initialize targets position
        self.targets = []
        for i in range(params['num_targets']):
            self.targets.append([random.randint(0,params['grid_size']['x']-1),random.randint(0,params['grid_size']['y']-1)])
            self.grid[self.targets[i][0],self.targets[i][1],2] = 1



instance=gridWorld({'grid_size':{'x':24,'y':24},'num_targets':5,'sub_groups':{'side_division':3}})