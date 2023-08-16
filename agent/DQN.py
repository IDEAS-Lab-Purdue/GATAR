import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from model.GAT_ml import GATPlanner
# Import the QNetwork
from agent.team import MRS 

# Define the DQN agent
class DQNAgent:
    def __init__(self, config):
        
        # Initialize the agent
        self.config = config
        self.action_dim = config['RL']['action_dim']
        self.learning_rate = config['RL']['learning_rate']
        self.gamma = config['RL']['gamma']
        self.tau = config['RL']['tau']
        self.buffer_size = config['RL']['buffer_size']
        self.batch_size = config['RL']['batch_size']
        self.epsilon = config['RL']['epsilon']
        self.weight_decay = config['RL']['weight_decay']
        self.device = config['device']

        # Q-Networks
        self.q_network = GATPlanner(config).to(self.device)
        if config['network']['load_pretrained'] is not None:
            self.q_network.load_state_dict(torch.load(config['network']['load_pretrained']))
            print('load pretrained model from {}'.format(config['network']['load_pretrained']))        
        self.target_network = GATPlanner(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.99)
        # Replay buffer
        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple("Experience", field_names=["obs","SList", "action", "reward", "next_obs","next_SList", "done"])

    def store_transition(self, obs,SList, action, reward, next_obs,next_SList, done):
        e = self.experience(obs,SList, action, reward, next_obs,next_SList, done)
        self.memory.append(e)

    def select_action(self, obs,SList, team, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.uniform(0, 1) > epsilon:
            
            self.q_network.add_graph(SList)
            obs=obs.to(self.config['device'])
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(obs) # B*N*A
                # choose action base on the probability
                action_values = action_values.reshape(-1,self.action_dim)
                action_values = torch.multinomial(torch.softmax(action_values,dim=1),1)
                action_values = action_values.reshape(-1,team.agent_num).cpu().numpy()
                
                
            self.q_network.train()
            return action_values
        else:
            # create a random int array 1*agent_num
            random_action = np.random.randint(self.action_dim, size=team.agent_num)
            return random_action


    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0

        experiences = random.sample(self.memory, k=self.batch_size)
        obs,SList, actions, rewards, next_obs,next_SList, dones = zip(*experiences)


        batch_obs = torch.from_numpy(np.vstack(obs)).float().to(self.device) # B*N*C*H*W
        batch_SList = torch.from_numpy(np.vstack(SList)).float().to(self.device) # B*K*N*N
        batch_actions = torch.from_numpy(np.vstack(actions)).long().to(self.device) # B*N
        batch_rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device).squeeze(1) # B*1
        batch_next_obs = torch.from_numpy(np.vstack(next_obs)).float().to(self.device)
        batch_next_SList = torch.from_numpy(np.vstack(next_SList)).float().to(self.device)
        batch_dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device).squeeze(1) # B*1


        
        # add graph
        self.q_network.add_graph(batch_SList)
        self.target_network.add_graph(batch_next_SList)

        Q_targets_next = self.target_network(batch_next_obs).detach() #B*N*A
        Q_targets_next = torch.sum(torch.max(Q_targets_next, dim=2)[0],dim=1)#B*N*1

        Q_targets = batch_rewards+ (self.gamma * Q_targets_next * (1 - batch_dones))

        
        Q_expected = self.q_network(batch_obs)
        Q_expected=torch.sum(Q_expected.gather(2, batch_actions.unsqueeze(2)).squeeze(2),dim=1)
        

        # Loss
        loss = nn.functional.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Update target network
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()