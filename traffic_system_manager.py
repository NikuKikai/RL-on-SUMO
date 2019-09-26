from rl_agents import Fixed_Q_Targets_Agent
import numpy as np
import matplotlib.pyplot as plt

class TrafficSystemManager:
    def __init__(self, road_structure, rl_args, verbose=True):
        self.agents_dict = {}
        self.cumulative_rewards_dict = {}
        self.rewards_dict = {}
        for intersection_name in road_structure:
            num_of_actions = road_structure[intersection_name]['num_of_phases']
            input_state_size = road_structure[intersection_name]['num_of_lanes'] * \
                               road_structure[intersection_name]['lane_len'] + \
                               num_of_actions

            # TODO: enable other agents too..
            self.agents_dict[intersection_name] = Fixed_Q_Targets_Agent(input_state_size, num_of_actions, rl_args, device='cuda')
            self.cumulative_rewards_dict[intersection_name] = []
            self.rewards_dict[intersection_name] = []

        if verbose:
            print('========== Traffic System Manager initialized with the following agents:')
            for key in self.agents_dict:
                print(key, ":", self.agents_dict[key])
            print('========================================================================')

    def select_action(self, state_dict):
        action_dict = {}
        for intersection_name in state_dict:
            state = state_dict[intersection_name]
            agent_name = intersection_name # for sake of explicity.
            action = self.agents_dict[agent_name].select_action(state)
            action_dict[intersection_name] = action
        return action_dict

    def add_to_memory(self, state_dict, action_dict, next_state_dict, reward_dict):
        for intersection_name in state_dict:
            state = state_dict[intersection_name]
            action = action_dict[intersection_name]
            next_state = next_state_dict[intersection_name]
            reward = reward_dict[intersection_name]

            agent_name = intersection_name  # for sake of explicity.
            self.agents_dict[agent_name].add_to_memory(state, action, next_state, reward)

            self.rewards_dict[agent_name].append(reward)

    def teach_agents(self):
        for agent_name in self.agents_dict:
            self.agents_dict[agent_name].optimize_model()

    def plot_learn_curve(self):
        for key in self.rewards_dict:
            self.cumulative_rewards_dict[key].append(np.mean(self.rewards_dict[key]))
            self.rewards_dict[key] = []

        for key in self.cumulative_rewards_dict:
            cumm_rewards = self.cumulative_rewards_dict[key]
            plt.plot(range(len(cumm_rewards)), cumm_rewards, label=key)
        plt.draw()
        plt.pause(0.001)
        plt.clf()
