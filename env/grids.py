
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
        self.sub_groups = []


        # randomly initialize targets position
        self.targets = []
        for i in range(params['num_targets']):
            self.targets.append([random.randint(0,params['grid_size']['x']-1),random.randint(0,params['grid_size']['y']-1)])
            self.grid[self.targets[i][0],self.targets[i][1],2] = 1



