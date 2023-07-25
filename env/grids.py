
import numpy as np
import random
import math
import matplotlib.pyplot as plt


# base class for environment

class baseEnv():

    def __init__(self):
        pass
        
    def reset(self):
        pass

    def step(self, action):
        pass

    def vis(self):
        pass

# extended class for grid world
class gridWorld(baseEnv):

    def __init__(self,params):
        super().__init__()
        self.params = params
        self.grid = np.zeros((params['grid_size']['x'],params['grid_size']['y'],3))
        # divided into subgroups
        self.num_subgroups = params['sub_groups']['side_division']**2
        self.subgroup_x_index = np.linspace(0,params['grid_size']['x'],params['sub_groups']['side_division']+1,dtype=int)
        self.subgroup_y_index = np.linspace(0,params['grid_size']['y'],params['sub_groups']['side_division']+1,dtype=int)
        
        

        # randomly initialize targets position
        
        
        # assign obstacles to each subgroup
        self.obstacles = []
        for i in range(self.params['sub_groups']['side_division']):
            for j in range(self.params['sub_groups']['side_division']):
                num_obstacles = random.randint(0,params['obstacles']['max_num'])
                for k in range(num_obstacles):
                    self.obstacles.append([random.randint(self.subgroup_x_index[i],self.subgroup_x_index[i+1]-1),
                                        random.randint(self.subgroup_y_index[j],self.subgroup_y_index[j+1]-1)])
                    self.grid[self.obstacles[-1][0],self.obstacles[-1][1],0] = 1

        self.targets = []
        for i in range(params['num_targets']):
            # randomly generate a target position that is not occupied by obstacles
            while True:
                position = [random.randint(0,params['grid_size']['x']-1),random.randint(0,params['grid_size']['y']-1)]
                if self.grid[position[0],position[1],0] == 0:
                    break
            self.targets.append(position)
            self.grid[self.targets[-1][0],self.targets[-1][1],2] = 1
        
        self.vis()


    def vis(self,filename=None):
        #visualize the grid
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks(np.arange(0,self.params['grid_size']['x']+1,1))
        self.ax.set_yticks(np.arange(0,self.params['grid_size']['y']+1,1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(True)
        self.ax.set_xlim(0,self.params['grid_size']['x'])
        self.ax.set_ylim(0,self.params['grid_size']['y'])
        self.ax.set_aspect('equal')
        
        # add different color for different subgroups
        for i in range(self.params['sub_groups']['side_division']):
            for j in range(self.params['sub_groups']['side_division']):
                self.ax.add_patch(plt.Rectangle((self.subgroup_x_index[i],self.subgroup_y_index[j]),
                                                self.subgroup_x_index[i+1]-self.subgroup_x_index[i],self.subgroup_y_index[j+1]-self.subgroup_y_index[j],
                                                fill=True,color=np.random.rand(3,1).flatten(),alpha=0.5))
                
        # indicate the targets with red dots
        for target in self.targets:
            self.ax.plot(target[0]+0.5,target[1]+0.5,'ro',markersize=1)
        
        # indicate the obstacles with black blocks
        for obstacle in self.obstacles:
            self.ax.add_patch(plt.Rectangle((obstacle[0],obstacle[1]),1,1,fill=True,color='k'))

        # save the figure
        if filename is not None:
            self.fig.savefig(filename)
        else:
            self.fig.savefig('grid.png')




