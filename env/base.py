# base class for environment

class baseEnv():

    def __init__(self):
        
        raise NotImplementedError
        
    def reset(self):
        
        raise NotImplementedError

    def step(self, action):
        
        raise NotImplementedError

    def vis(self):
        
        raise NotImplementedError