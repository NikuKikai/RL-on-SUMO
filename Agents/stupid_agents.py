import numpy as np
from random import randrange

class Stupid_Agent():
    def __init__(self, n_actions=4):
        self.n_actions = n_actions
        self.eps_threshold = 0

        # Performance buffers.
        self.rewards_list = []

    def optimize_model(self):
        return

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


class Random_Agent(Stupid_Agent):
    '''
    This agent only takes a random phase.
    '''
    def __init__(self, n_actions=4):
        super().__init__(n_actions)

    def select_action(self, state=None):
        return randrange(self.n_actions)


class Cyclic_Agent(Stupid_Agent):
    '''
    This agent only change the phases repeatedly in a cycle.
    '''

    def __init__(self, n_actions=4):
        super().__init__(n_actions)

        self.curr_action = 0
        self.curr_time = 0
        self.time_on_each_phase = 10


    def select_action(self, state=None):
        self.curr_time += 1
        if self.curr_time > self.time_on_each_phase:
            # Return next phase.
            self.curr_time = 0
            if self.curr_action + 1 < self.n_actions:
                self.curr_action += 1
                return self.curr_action
            else:
                self.curr_action = 0
                return self.curr_action
        else:
            # return the same phase
            return self.curr_action
