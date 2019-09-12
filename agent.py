import tensorflow as tf
import numpy as np
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

class fixed_target_Agent():
    '''
    this clas implements Fixed-q target agent with pytorch
    '''
    def __init__(self, state_size, n_actions, device='cpu'):
        # Get number of actions from gym action space
        self.device = device

        # Exploration / Exploitation params.
        self.steps_done = 0
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200

        # Learning params
        self.batch_size = 128
        self.discount = 0.999

        # Deep q networks
        self.policy_net = DQN(state_size, n_actions).to(device)
        self.target_net = DQN(state_size, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = self.exp_end + (self.exp_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = memory.sample(self.batch_size)
        # This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                      batch.next_state)), device=self.device,
                                      dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

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
        expected_state_action_values = (next_state_values * self.discount) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # clip gradients.
        # TODO: explain why
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class Agent:
    def __init__(self):
        self.lane_len = 10
        self.GAMMA = 0.9

        self.states = []
        self.rewards = []
        self.actions = []

        self.init_nn()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        return

    def init_nn(self):
        self.state_layer = tf.placeholder(tf.float32, [None, self.lane_len*8 + 4], 'state')
        self.lane_layers = [tf.slice(self.state_layer, [0, i*self.lane_len], [-1, self.lane_len]) for i in range(0, 8)]
        self.phase = tf.slice(self.state_layer, [0, 8*self.lane_len], [-1, 4])
        self.subsize = 4

        with tf.name_scope('subnet'):
            dim = [self.lane_len, 16, 8, self.subsize]
            nlayer = len(dim)-1
            w = [tf.Variable(tf.truncated_normal(dim[i:i + 2]) / dim[i]) for i in range(nlayer)]
            b = [tf.Variable(tf.constant(0.00, shape=[dim[i + 1]])) for i in range(nlayer)]
            for iw in w: tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, iw)
            for ib in b: tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, ib)
            layer = []
            for ilane in self.lane_layers:
                for ilayer in range(nlayer):
                    ilane = tf.matmul(ilane, w[ilayer]) + b[ilayer]
                layer.append(tf.nn.leaky_relu(ilane))
            self.sub_layers = []
            self.sub_layers.append(layer[0]+layer[4])
            self.sub_layers.append(layer[2]+layer[6])
            self.sub_layers.append(layer[1]+layer[5])
            self.sub_layers.append(layer[3]+layer[7])

        with tf.name_scope('actornet'):
            dim = [self.subsize, 16, 8, 4, 1]
            nlayer = len(dim)-1
            for n in range(4):
                w = [tf.Variable(tf.truncated_normal(dim[i:i + 2]) / dim[i]) for i in range(nlayer)]
                b = [tf.Variable(tf.constant(0.00, shape=[dim[i + 1]])) for i in range(nlayer)]
                for iw in w: tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, iw)
                for ib in b: tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, ib)
                layer4 = []
                for k in range(4):
                    layer = self.sub_layers[(k+n) % 4]
                    for ilayer in range(nlayer-1):
                        layer = tf.matmul(layer, w[ilayer]) + b[ilayer]
                        layer = tf.nn.leaky_relu(layer)
                    layer = tf.matmul(layer, w[-1]) + b[-1]
                    layer4.append(layer)
                if n < 1:
                    self.action_layer = tf.concat(layer4, 1)
                else:
                    self.action_layer = self.action_layer + tf.concat(layer4, 1)

            dim = [4, 16, 8, 4]
            nlayer = len(dim) - 1
            w = [tf.Variable(tf.truncated_normal(dim[i:i + 2]) / dim[i]) for i in range(nlayer)]
            b = [tf.Variable(tf.constant(0.00, shape=[dim[i + 1]])) for i in range(nlayer)]
            for iw in w: tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, iw)
            for ib in b: tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, ib)
            layer = self.phase
            for ilayer in range(nlayer):
                layer = tf.nn.leaky_relu(tf.matmul(layer, w[ilayer])+b[ilayer])
            # self.action_layer += layer
            self.action_layer = tf.nn.softmax(self.action_layer)

            self.advantage_fb = tf.placeholder(tf.float32, [None])
            self.action_fb = tf.placeholder(tf.float32, [None, 4])
            p = tf.reduce_mean(tf.multiply(self.action_layer, self.action_fb), reduction_indices=1)
            logp = tf.log(tf.clip_by_value(p, 1e-8, 1.))
            cost = - tf.reduce_mean(tf.multiply(self.advantage_fb, logp))

            regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'actornnet')
            reg_variables.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'subnet'))
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            cost += reg_term

            self.lr = tf.placeholder(tf.float32)
            self.actor_opt = tf.train.AdamOptimizer(self.lr).minimize(cost)

        with tf.name_scope('criticnet'):
            dim = [self.subsize, 16, 8, 4, 1]
            nlayer = len(dim) - 1
            w = [tf.Variable(tf.truncated_normal(dim[i:i+2]) / dim[i]) for i in range(nlayer)]
            b = [tf.Variable(tf.constant(0.00, shape=[dim[i+1]])) for i in range(nlayer)]
            for iw in w: tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, iw)
            for ib in b: tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, ib)
            layer4 = []
            for isub in self.sub_layers:
                for ilayer in range(nlayer-1):
                    isub = tf.matmul(isub, w[ilayer]) + b[ilayer]
                    isub = tf.nn.leaky_relu(isub)
                layer4.append(tf.matmul(isub, w[-1]) + b[-1])
            self.value_layer = layer4[0] + layer4[1] + layer4[2] + layer4[3]

            self.return_fb = tf.placeholder(tf.float32, [None, 1])
            cost = tf.losses.mean_squared_error(self.return_fb, self.value_layer)

            regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'criticnet')
            reg_variables.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'subnet'))
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            cost += reg_term

            self.critic_opt = tf.train.AdamOptimizer(self.lr).minimize(cost)

    def _train(self, advantage, Return, action, state, lr, iter):
        for _ in range(iter[0]):
            self.actor_opt.run(feed_dict={
                self.advantage_fb: advantage,
                self.state_layer: state,
                self.action_fb: action,
                self.lr: lr
                })
        for _ in range(iter[1]):
            self.critic_opt.run(feed_dict={
                self.return_fb: Return,
                self.state_layer: state,
                self.lr: lr
                })

    def policy(self, state):
        y = self.sess.run(self.action_layer, feed_dict={self.state_layer: [state]})
        action = np.random.choice(4, p=y[0])
        return action

    def value(self, state):
        y = self.sess.run(self.value_layer, feed_dict={self.state_layer: [state]})
        return np.squeeze(y)

    def train(self, state, action, reward, lr, para):
        self.states.append(state)
        self.rewards.append(reward)
        a = np.eye(4)[action]
        self.actions.append(a)
        [TDn, batch, trig, iter0, iter1] = para
        iter = [iter0, iter1]

        length = len(self.rewards)
        if trig or length > batch * TDn:
            r = np.array(self.rewards)
            returns = np.zeros_like(r)
            values = np.zeros_like(r)
            discounted_sum = self.value(state)
            gammas = np.hstack([self.GAMMA**n for n in range(0, TDn)])

            for t in reversed(range(0, length)):
                if t > (length-TDn-2):
                    discounted_sum = discounted_sum * self.GAMMA + r[t]
                    returns[t] = discounted_sum
                else:
                    returns[t] = np.sum(gammas * r[t:(t+TDn)]) + self.value(self.states[t+TDn])

                values[t] = self.value(self.states[t])

            advantage = returns - values

            actions = np.array(self.actions)
            returns = np.reshape(returns, [length, 1])
            self._train(advantage, returns, actions, self.states, lr, iter)

            self.rewards, self.actions, self.states = [], [], []
