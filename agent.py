import tensorflow as tf
import numpy as np

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
