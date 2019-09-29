from deep_q_network import DQN

import math
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from replay_memory import Transition
from replay_memory import ReplayMemory
from datetime import datetime
from random import randrange
import utils


class Random_Agent():
    '''
    This agent only takes a random phase.
    '''
    def __init__(self, n_actions=4):
        self.n_actions = n_actions

        # Performance buffers.
        self.rewards_list = []

    def select_action(self, state=None):
        return randrange(self.n_actions)

    def add_to_memory(self, state, action, next_state, reward):
        # This method used only for saving the reward.
        self.rewards_list.append(reward)

    def dump_rewards(self):
        '''
        Save cumulative rewards to file
        '''
        # TODO: write smarter logging interface.
        with open('random_agent_dump.txt', 'a') as file:
            file.write(str(np.mean(np.array(self.rewards_list))) + '\n')
        self.rewards_list = []

class Cyclic_Agent():
    '''
    This agent only change the phases repeatedly in a cycle.
    '''
    def __init__(self, n_actions=4):
        self.n_actions = n_actions
        self.curr_action = 0
        self.curr_time = 0
        self.time_on_each_phase = 10

        # Performance buffers.
        self.rewards_list = []

    def select_action(self, state=None):
        self.curr_time += 1
        if self.curr_time > self.time_on_each_phase:
            # Return next phase.
            self.curr_time = 0
            if self.curr_action+1 < self.n_actions:
                self.curr_action += 1
                return self.curr_action
            else:
                self.curr_action = 0
                return self.curr_action
        else:
            # return the same phase
            return self.curr_action

    def add_to_memory(self, state, action, next_state, reward):
        # This method used only for saving the reward.
        self.rewards_list.append(reward)

    def dump_rewards(self):
        '''
        Save cumulative rewards to file
        '''
        # TODO: write smarter logging interface.
        with open('cyclic_agent_dump.txt', 'a') as file:
            file.write(str(np.mean(np.array(self.rewards_list))) + '\n')
        self.rewards_list = []

class RL_Agent():
    def __init__(self, state_size, n_actions, args, device='cpu'):
        # Get number of actions from gym action space
        self.device = device

        # Exploration / Exploitation params.
        self.steps_done = 0
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay

        # RL params
        self.target_update = args.target_update
        self.discount = args.discount

        # Env params
        self.n_actions = n_actions
        self.state_size = state_size

        # Deep q networks params
        self.batch_size = args.batch_size
        self.policy_net = DQN(state_size, n_actions).to(self.device)

        if str(args.optimizer).lower() == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters())
        if str(args.optimizer).lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.policy_net.parameters())
        else:
            raise NotImplementedError

        self.memory = ReplayMemory(args.replay_size)

        # Performance buffers.
        self.rewards_list = []

        # Logging params
        self.cumulative_rewards_log = str(datetime.timestamp(datetime.now())).split('.')[0] \
                                      + '_' + args.output_file_name

    def dump_rewards(self):
        '''
        Save cumulative rewards to file
        '''
        # TODO: write smarter logging interface.
        with open(self.cumulative_rewards_log, 'a') as file:
            file.write(str(np.mean(np.array(self.rewards_list))) + '\n')
        self.rewards_list = []

    def add_to_memory(self, state, action, next_state, reward):
        self.rewards_list.append(reward)
        state = torch.from_numpy(state)
        action = torch.tensor([action])
        next_state = torch.from_numpy(next_state)
        reward = torch.tensor([reward])
        self.memory.push(state, action, next_state, reward)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        #print('eps_threshold=', eps_threshold)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = torch.from_numpy(state).to(self.device) # Convert to tensor.
                state = state.unsqueeze(0) # Add batch dimeniton.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long).item()



class DQN_Agent(RL_Agent):
    '''
    Regular Q-Learning Agent
    One deep network.
    DQN - to predict Q of a given action, value a state. i.e. Q(s,a) and Q(s', a') for loss calculation.
    '''
    def __init__(self, state_size, n_actions, args, device):
        RL_Agent.__init__(self, state_size, n_actions, args, device=device)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                      batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).view(self.batch_size,-1).to(self.device)
        state_batch = torch.cat(batch.state).view(self.batch_size,-1).to(self.device)
        action_batch = torch.cat(batch.action).view(self.batch_size,-1).to(self.device)
        reward_batch = torch.cat(batch.reward).view(self.batch_size,-1).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states using the same policy net.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.policy_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.discount) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # clip gradients.
        # TODO: explain why
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Saving a checkpoint is done in the following way :
        # Should we consider adding another fields?
        # define is_best variable to indicate weather the current agent is the best so far
        # utils.save_checkpoint({'epoch': should this be saved? what is epoch?,
        #                         'policy_net': self.policy_net.state_dict(),
        #                         'optimizer' : self.optimizer,
        #                         'train_loss': loss}
        #                         is_best =is_best,
        #                         checkpoint="weights_and_val")


class Fixed_Q_Targets_Agent():
    '''
    This agent implements Fixed Q-Targets algorithm. There are two deep networks.
    Policy network - to predict Q of a given action, value a state. i.e. Q(s,a)
    Target network - to predict Q values of action of the next state. i.e. max Q(s', a') for loss calculation.
    '''
    def __init__(self, state_size, n_actions, args, device='cpu'):
        # Get number of actions from gym action space
        self.device = device

        # Exploration / Exploitation params.
        self.steps_done = 0
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay

        # RL params
        self.target_update = args.target_update
        self.discount = args.discount

        # Env params
        self.n_actions = n_actions
        self.state_size = state_size

        # Deep q networks params
        self.batch_size = args.batch_size
        self.policy_net = DQN(state_size, n_actions).to(self.device)
        self.target_net = DQN(state_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if str(args.optimizer).lower() == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters())
        if str(args.optimizer).lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.policy_net.parameters())
        else:
            raise NotImplementedError

        self.memory = ReplayMemory(args.replay_size)

        # Performance buffers.
        self.rewards_list = []

        # Logging params
        self.cumulative_rewards_log = str(datetime.timestamp(datetime.now())).split('.')[0] \
                                      + '_' + args.output_file_name

    def dump_rewards(self):
        '''
        Save cumulative rewards to file
        '''
        # TODO: write smarter logging interface.
        with open(self.cumulative_rewards_log, 'a') as file:
            file.write(str(np.mean(np.array(self.rewards_list))) + '\n')
        self.rewards_list = []

    def add_to_memory(self, state, action, next_state, reward):
        self.rewards_list.append(reward)
        state = torch.from_numpy(state)
        action = torch.tensor([action])
        next_state = torch.from_numpy(next_state)
        reward = torch.tensor([reward])
        self.memory.push(state, action, next_state, reward)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        #print('eps_threshold=', eps_threshold)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = torch.from_numpy(state).to(self.device) # Convert to tensor.
                state = state.unsqueeze(0) # Add batch dimeniton.
                action = self.policy_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long).item()

        return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                      batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).view(self.batch_size,-1).to(self.device)
        state_batch = torch.cat(batch.state).view(self.batch_size,-1).to(self.device)
        action_batch = torch.cat(batch.action).view(self.batch_size,-1).to(self.device)
        reward_batch = torch.cat(batch.reward).view(self.batch_size,-1).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.discount) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # clip gradients.
        # TODO: explain why
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network, copying all weights and biases in DQN
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


        # Saving a checkpoint is done in the following way :
        # Should we consider adding another fields?
        # define is_best variable to indicate weather the current agent is the best so far
        # utils.save_checkpoint({'epoch': should this be saved? what is epoch?,
        #                         'policy_net': self.policy_net.state_dict(),
        #                         'target_net': self.target_net.state_dict(),
        #                         'optimizer' : self.optimizer,
        #                         'train_loss': loss}
        #                         is_best =is_best,
        #                         checkpoint="weights_and_val")


class Double_DQN_Agent():
    '''
    Double DQN Agents
    DQN - to predict Q of a given action, value a state. i.e. Q(s,a) and the  a' = argmax Q(s', a)
    Target - used to calc Q(s', a') for loss calculation.
    '''

class Dueling_DQN_Agent():
    '''
    Dueling Double DQN Agents
    '''