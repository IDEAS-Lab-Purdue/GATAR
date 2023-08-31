import numpy as np
import random
import matplotlib.pyplot as plt
from env.base import baseEnv
import os
from tcod import libtcodpy
import tcod
import torch


# extended class for grid world
class gridWorld(baseEnv):

    def __init__(self,params,map_dict,verbose=False):
        
        self.params = params
        self.grid = np.zeros((params['grid_size']['x'],params['grid_size']['y'],3))
        self.agents_observation=np.zeros((params['num_agents'],params['grid_size']['x'],params['grid_size']['y'],3))
        self.agents_pos = np.zeros((params['num_agents'],2),dtype=int)
        obstacles = map_dict['obstacles'] 
        if len (obstacles)!=0:
            self.obstacles = np.array(obstacles)
            self.grid[self.obstacles[:,0],self.obstacles[:,1],0] = 1
        else:
            self.obstacles = None
        targets=map_dict['targets']
        self.targets = np.array(targets)
        self.grid[self.targets[:,0],self.targets[:,1],2] = 1
        #self.vis(store=True)
        if verbose:
            print('Environment initialized')
            print("Targets' positions: ",self.targets.shape)
            if self.obstacles is not None:
                print("Obstacles' positions: ",self.obstacles.shape)
            else:
                print("No obstacles")
            print("Agents' positions: ",self.agents_pos.shape)
            print('------------------------')

    
    def sync(self):
        
        self.grid[:,:,1]=0
        for i in range(self.agents_pos.shape[0]):
            self.grid[self.agents_pos[i,0],self.agents_pos[i,1],1] = 1
        self.grid[:,:,2]=0
        for i in range(self.targets.shape[0]):
            self.grid[self.targets[i,0],self.targets[i,1],2] = 1
        


    def vis(self,obs,draw_arrows=False,store=False,filename=None):
        #visualize the grid
        self.fig = plt.figure()
        mode=self.params['vis']['mode']
        if mode=='global&local':
            self.ax = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)
        elif mode=='global':
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
        # for i in range(self.params['sub_groups']['side_division']):
        #     for j in range(self.params['sub_groups']['side_division']):
        #         self.ax.add_patch(plt.Rectangle((self.subgroup_x_index[i],self.subgroup_y_index[j]),
        #                                         self.subgroup_x_index[i+1]-self.subgroup_x_index[i],self.subgroup_y_index[j+1]-self.subgroup_y_index[j],
        #                                         fill=True,color=np.random.rand(3,1).flatten(),alpha=0.5))
                
        # indicate the targets with red dots
        for i in range(self.targets.shape[0]):
            self.ax.plot(self.targets[i,0]+0.5,self.targets[i,1]+0.5,'ro',markersize=1)
        
        # indicate the obstacles with black blocks
        if self.obstacles is not None:
            for i in range(self.obstacles.shape[0]):
                self.ax.add_patch(plt.Rectangle((self.obstacles[i,0],self.obstacles[i,1]),1,1,fill=True,color='k'))

        # indicate the agents with blue dots
        for i in range(self.agents_pos.shape[0]):

            self.ax.plot(self.agents_pos[i,0]+0.5,self.agents_pos[i,1]+0.5,'bo',markersize=1)

        
    
        #make the local observation area of agent 0 yellow
        if mode=='global&local':
            vis_agent_id=0
            observation_test=obs[0]
            # add subplot to draw the observation area
            self.ax2.set_xticks(np.arange(0,self.params['grid_size']['x']+1,1))
            self.ax2.set_yticks(np.arange(0,self.params['grid_size']['y']+1,1))
            self.ax2.set_xticklabels([])
            self.ax2.set_yticklabels([])
            self.ax2.grid(True)
            self.ax2.set_xlim(0,self.params['grid_size']['x'])
            self.ax2.set_ylim(0,self.params['grid_size']['y'])
            self.ax2.set_aspect('equal')
            # draw observed obstacles
            local_obstacles = np.where(observation_test[:,:,0]==1)
            for i in range(len(local_obstacles[0])):
                self.ax2.add_patch(plt.Rectangle((local_obstacles[0][i],local_obstacles[1][i]),1,1,fill=True,color='k'))
            # draw observed targets
            local_targets = np.where(observation_test[:,:,2]==1)
            for i in range(len(local_targets[0])):
                self.ax2.plot(local_targets[0][i]+0.5,local_targets[1][i]+0.5,'ro',markersize=1)
            # add a dim yellow background on the observed area
            observed_area = np.where(observation_test[:,:,0]<0)
            for i in range(len(observed_area[0])):
                self.ax2.add_patch(plt.Rectangle((observed_area[0][i],observed_area[1][i]),1,1,fill=True,color='black',alpha=0.1))
            # draw the ego agent
            self.ax2.plot(self.agents_pos[vis_agent_id,0]+0.5,self.agents_pos[vis_agent_id,1]+0.5,'b^',markersize=1)
            
            vis_agent_id=-1
            observation_test=obs[vis_agent_id]
            # add subplot to draw the observation area
            self.ax3.set_xticks(np.arange(0,self.params['grid_size']['x']+1,1))
            self.ax3.set_yticks(np.arange(0,self.params['grid_size']['y']+1,1))
            self.ax3.set_xticklabels([])
            self.ax3.set_yticklabels([])
            self.ax3.grid(True)
            self.ax3.set_xlim(0,self.params['grid_size']['x'])
            self.ax3.set_ylim(0,self.params['grid_size']['y'])
            self.ax3.set_aspect('equal')
            # draw observed obstacles
            local_obstacles = np.where(observation_test[:,:,0]==1)
            for i in range(len(local_obstacles[0])):
                self.ax3.add_patch(plt.Rectangle((local_obstacles[0][i],local_obstacles[1][i]),1,1,fill=True,color='k'))
            # draw observed targets
            local_targets = np.where(observation_test[:,:,2]==1)
            for i in range(len(local_targets[0])):
                self.ax3.plot(local_targets[0][i]+0.5,local_targets[1][i]+0.5,'ro',markersize=1)
            # add a dim yellow background on the observed area
            observed_area = np.where(observation_test[:,:,0]<0)
            for i in range(len(observed_area[0])):
                self.ax3.add_patch(plt.Rectangle((observed_area[0][i],observed_area[1][i]),1,1,fill=True,color='black',alpha=0.1))
            # draw the ego agent
            self.ax3.plot(self.agents_pos[vis_agent_id,0]+0.5,self.agents_pos[vis_agent_id,1]+0.5,'b^',markersize=1)



        # save the figure
        if store:
            if filename is not None:
                self.fig.savefig(filename)
            else:
                self.fig.savefig('grid.png')
        
        # turn it into numpy array
        self.fig.canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(self.fig)
        
        return data

    def step(self,agents,action):
        
         
        directions = np.array([[0,0],[0, 1], [0, -1], [1, 0], [-1, 0],[1,1],[1,-1],[-1,1],[-1,-1]])

        # update agents' position
        
        new_agents_pos = agents.agents_pos+ directions[action.ravel()]

        # check whether the agents are out of bound
        is_out_of_bound_x= np.logical_or(new_agents_pos[:, 0] < 0, new_agents_pos[:, 0] >= self.params['grid_size']['x'])
        is_out_of_bound_y= np.logical_or(new_agents_pos[:, 1] < 0, new_agents_pos[:, 1] >= self.params['grid_size']['y'])
        is_out_of_bound = is_out_of_bound_x | is_out_of_bound_y

        print("is_out_of_bound: {}".format(is_out_of_bound.shape))
        print("new_agents_pos: {}".format(new_agents_pos.shape))
        print("agents.agents_pos: {}".format(agents.agents_pos.shape))
        new_agents_pos[is_out_of_bound] = agents.agents_pos[is_out_of_bound]

        new_agents_pos = new_agents_pos.astype(int)
        # check whether the agents hit obstacles
        is_hit_obstacles = self.grid[new_agents_pos[:, 0], new_agents_pos[:, 1], 0] == 1
        
        
        
        for i in range(len(agents.agents_type)):
            if agents.agents_type[i]=="UAV":
                is_hit_obstacles[i]=False  
        # make the conflicting agents stay still
        if np.sum(is_hit_obstacles)>0:
            new_agents_pos[is_hit_obstacles] = self.agents_pos[is_hit_obstacles]
        # check whether the agents reach the targets
        is_reach_target = self.grid[new_agents_pos[:, 0], new_agents_pos[:, 1], 2] == 1
        reward = np.sum(is_reach_target)-0.1

        # remove the targets that are reached
        self.grid[new_agents_pos[is_reach_target, 0], new_agents_pos[is_reach_target, 1], 2] = 0
        # update the targets list
        self.targets=np.array(np.where(self.grid[:,:,2]==1)).T
        done=False
        if self.targets.shape[0]==0:
            done=True
        self.agents_pos = new_agents_pos
        agents.agents_pos = new_agents_pos.astype(int)
        # update the grid
        self.grid[:,:,1]=0
        for i in range(len(agents.agents_type)):
            self.grid[self.agents_pos[i,0],self.agents_pos[i,1],1] = 1
        
        return reward,done
            
        
    def copy(self):
        new_instance = gridWorld(self.params,self.obstacles)
        new_instance.grid = self.grid.copy()
        new_instance.agents_pos = self.agents_pos.copy()
        new_instance.targets = self.targets.copy()
        return new_instance