from Agents.DQN import DQN
import math
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.replay_memory import Transition, ReplayMemory
from datetime import datetime

import os


class DQN_Agent():
    '''
    Regular Q-Learning Agent
    One deep network.
    DQN - to predict Q of a given action, value a state. i.e. Q(s,a) and Q(s', a') for loss calculation.
    '''
    def __init__(self, state_size, n_actions, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device

        # Exploration / Exploitation params.
        self.steps_done = 0
        self.eps_threshold = 1
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
        self.layers = args.layers
        self.batch_size = args.batch_size
        self.policy_net = DQN(state_size, n_actions, layers=self.layers).to(self.device).float()
        self.target_net = None
        self.grad_clip = args.grad_clip

        if str(args.optimizer).lower() == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters())
        if str(args.optimizer).lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.policy_net.parameters())
        else:
            raise NotImplementedError

        self.memory = ReplayMemory(args.replay_size)

        # Performance buffers.
        self.rewards_list = []

    def add_to_memory(self, state, action, next_state, reward):
        self.rewards_list.append(reward)
        state = torch.from_numpy(state).float()
        action = torch.tensor([action])
        next_state = torch.from_numpy(next_state).float()
        reward = torch.tensor([reward])
        self.memory.push(state, action, next_state, reward)

    def select_action(self, state):
        sample = random.random()
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = torch.from_numpy(state).float().to(self.device) # Convert to tensor.
                state = state.unsqueeze(0) # Add batch dimension.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long).item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        next_states_batch = torch.cat(batch.next_state).view(self.batch_size, -1).to(self.device)
        state_batch = torch.cat(batch.state).view(self.batch_size, -1).to(self.device)
        action_batch = torch.cat(batch.action).view(self.batch_size, -1).to(self.device)
        reward_batch = torch.cat(batch.reward).view(self.batch_size, -1).to(self.device)

        # Compute loss
        loss = self._compute_loss(state_batch, action_batch, next_states_batch, reward_batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # clip grad
        if self.grad_clip is not None:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-self.grad_clip, self.grad_clip)

        # update Policy net weights
        self.optimizer.step()

        # update Target net weights
        self._update_target()

    def _compute_loss(self, state_batch, action_batch, next_states_batch, reward_batch):
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states using the same policy net.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values = self.policy_net(next_states_batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.discount) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        return loss

    def _update_target(self):
        if self.target_net is None:
            # There is nothing to update.
            return

        # Update the target network, copying all weights and biases in DQN
        if self.target_update > 1:
            # Hard copy of weights.
            if self.steps_done % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            return
        elif self.target_update < 1 and self.target_update > 0:
            # polyak averaging:
            tau = self.target_update
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(tau * param + (1 - tau) * target_param)
            return
        else:
            raise NotImplementedError

    def save_ckpt(self, ckpt_folder):
        '''
        saves checkpoint of policy net in ckpt_folder
        :param ckpt_folder: path to a folder.
        '''
        ckpt_path = os.path.join(ckpt_folder, 'policy_net_state_dict.pth')
        torch.save(self.policy_net.state_dict(), ckpt_path)

class Fixed_Q_Targets_Agent(DQN_Agent):
    '''
    This agent implements Fixed Q-Targets algorithm. There are two deep networks.
    Policy network - to predict Q of a given action, value a state. i.e. Q(s,a)
    Target network - to predict Q values of action of the next state. i.e. max Q(s', a') for loss calculation.
    '''
    def __init__(self, state_size, n_actions, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(state_size, n_actions, args, device=device)
        self.target_net = DQN(state_size, n_actions, layers=self.layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def _compute_loss(self, state_batch, action_batch, next_states_batch, reward_batch):
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
        next_state_values = self.target_net(next_states_batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.discount) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        return loss

class Double_DQN_Agent(DQN_Agent):
    '''
    Double DQN Agent. Sourses:
    https://towardsdatascience.com/double-deep-q-networks-905dd8325412
    https://www.freecodecamp.org/news/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682/

    Denote 'Policy net' (forward + backward) and 'Target net' (forward only).
    During optimization we do 3 forward passes.
    1) Policy net: Q(state, action) to get Q values of state, action pairs.
    2) Policy net: Q(next_state, ) to predict the next_action
    3) Target net: Q(next_state, next_action) to get Q values of next_state, next_action pairs.
    ***NOTICE: in some sources, in step 2) the use target net, and in step 3) use policy net.
    Then compute loss and do backward step on Policy net.
    Copy with polyak averaging weights from policy net into target net.
    '''
    def __init__(self, state_size, n_actions, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(state_size, n_actions, args, device=device)
        self.target_net = DQN(state_size, n_actions, layers=self.layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def _compute_loss(self, state_batch, action_batch, next_states_batch, reward_batch):
        # Q{policy net}(s, a)
        state_action_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # argmax{a} Q{policy net}(s', a')
        next_state_actions = torch.argmax(self.policy_net(next_states_batch), dim=1).unsqueeze(1)

        # Q{ploicy net}(s', argmax{a} Q{target net}(s', a') )
        next_state_q_values = self.target_net(next_states_batch).gather(1, next_state_actions)

        # Q* = Disount * Q(s', argmax(..)) + R
        expected_state_action_values = (next_state_q_values * self.discount) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_q_values, expected_state_action_values)
        return loss
