
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
                if 'group_obstacle_matrix' not in params['sub_groups']:
                    num_obstacles = random.randint(0,params['obstacles']['max_num'])
                else:
                    num_obstacles = params['sub_groups']['group_obstacle_matrix'][i][j]
                for k in range(num_obstacles):
                    self.obstacles.append([random.randint(self.subgroup_x_index[i],self.subgroup_x_index[i+1]-1),
                                        random.randint(self.subgroup_y_index[j],self.subgroup_y_index[j+1]-1)])
                    self.grid[self.obstacles[-1][0],self.obstacles[-1][1],0] = 1
        
        self.obstacles = np.array(self.obstacles)

        self.__assign_targets_agents()
        

        self.vis(store=True)
        print('Environment initialized')
        print("Targets' positions: ",self.targets.shape)
        print("Obstacles' positions: ",self.obstacles.shape)
        print("Agents' positions: ",self.agents.shape)
        print('------------------------')



    def vis(self,draw_arrows=False,store=False,filename=None):
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
        
        # # add different color for different subgroups
        # for i in range(self.params['sub_groups']['side_division']):
        #     for j in range(self.params['sub_groups']['side_division']):
        #         self.ax.add_patch(plt.Rectangle((self.subgroup_x_index[i],self.subgroup_y_index[j]),
        #                                         self.subgroup_x_index[i+1]-self.subgroup_x_index[i],self.subgroup_y_index[j+1]-self.subgroup_y_index[j],
        #                                         fill=True,color=np.random.rand(3,1).flatten(),alpha=0.5))
                
        # indicate the targets with red dots
        for i in range(self.targets.shape[0]):
            self.ax.plot(self.targets[i,0]+0.5,self.targets[i,1]+0.5,'ro',markersize=1)
        
        # indicate the obstacles with black blocks
        for i in range(self.obstacles.shape[0]):
            self.ax.add_patch(plt.Rectangle((self.obstacles[i,0],self.obstacles[i,1]),1,1,fill=True,color='k'))

        # indicate the agents with blue triangles
        for i in range(self.agents.shape[0]):
            self.ax.plot(self.agents[i,0]+0.5,self.agents[i,1]+0.5,'b^',markersize=1)
            if draw_arrows:
                self.ax.arrow(self.old_agents[i,0]+0.5,self.old_agents[i,1]+0.5,
                                self.agents[i,0]-self.old_agents[i,0],self.agents[i,1]-self.old_agents[i,1],
                                head_width=0.1, head_length=0.1, fc='k', ec='k')


        #save the figure
        if store:
            if filename is not None:
                self.fig.savefig(filename)
            else:
                self.fig.savefig('grid.png')
        
        # turn it into numpy array
        self.fig.canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # close the figure
        plt.close(self.fig)
        # return the np array
        return data

    def step(self,action):
        
        directions = np.array([[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]])

        # update agents' position
        new_agents = self.agents + directions[action.ravel()]

        # check whether the agents are out of bound
        is_out_of_bound_x= np.logical_or(new_agents[:, 0] < 0, new_agents[:, 0] >= self.params['grid_size']['x'])
        is_out_of_bound_y= np.logical_or(new_agents[:, 1] < 0, new_agents[:, 1] >= self.params['grid_size']['y'])
        is_out_of_bound = is_out_of_bound_x | is_out_of_bound_y

        # check whether the agents hit obstacles
        
        is_hit_obstacles = self.grid[new_agents[:, 0], new_agents[:, 1], 0] == 1

        # make the conflicting agents stay still
        new_agents[is_out_of_bound | is_hit_obstacles] = self.agents[is_out_of_bound | is_hit_obstacles]
        

        # check whether the agents reach the targets
        is_reach_target = self.grid[new_agents[:, 0], new_agents[:, 1], 2] == 1
        

        # remove the targets that are reached
        self.grid[new_agents[is_reach_target, 0], new_agents[is_reach_target, 1], 2] = 0
        
        # update the targets list

        self.targets=np.array(np.where(self.grid[:,:,2]==1)).T

        self.old_agents=self.agents
        self.agents = new_agents

    def __assign_targets_agents(self):
        self.targets = []
        for i in range(self.params['num_targets']):
            # randomly generate a target position that is not occupied by obstacles
            while True:
                position = [random.randint(0,self.params['grid_size']['x']-1),random.randint(0,self.params['grid_size']['y']-1)]
                if self.grid[position[0],position[1],0] == 0:
                    break
            self.targets.append(position)
            self.grid[self.targets[-1][0],self.targets[-1][1],2] = 1
        self.targets = np.array(self.targets)
        self.agents = np.zeros((self.params['num_agents'],2),dtype=int)
        self.old_agents = np.zeros((self.params['num_agents'],2),dtype=int)


    def reset(self):
        # restore agents and targets keep obstacles
        self.grid[:,:,2]=0
        self.__assign_targets_agents()
