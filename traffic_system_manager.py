from rl_agents import Fixed_Q_Targets_Agent
import numpy as np
import matplotlib.pyplot as plt

class TrafficSystemManager:
    def __init__(self, dim_dict, rl_args, verbose=True):
        self.agents_dict = {}

        # Metrics logging:
        self.avg_rewards_dict = {} # Average reward per episode
        self.rewards_dict = {} # Rewards of an episode
        self.cumulative_rewards_dict = {} # Cumulative rewards of entire experiment

        for intersection_name in dim_dict:
            input_state_size, num_of_actions = dim_dict[intersection_name]
            # TODO: enable other agents too..
            self.agents_dict[intersection_name] = Fixed_Q_Targets_Agent(input_state_size, num_of_actions, rl_args, device='cuda')

            # Initialize all loggers.
            self.avg_rewards_dict[intersection_name] = []
            self.rewards_dict[intersection_name] = []
            self.cumulative_rewards_dict[intersection_name] = [0]

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
            last_cumulative_reward = self.cumulative_rewards_dict[agent_name][-1]
            self.cumulative_rewards_dict[agent_name].append(last_cumulative_reward + reward)

    def teach_agents(self):
        for agent_name in self.agents_dict:
            self.agents_dict[agent_name].optimize_model()

    def dump_learn_curve(self, plot=False):
        '''
        This method saves the average reward of the performed steps up to this point.
        Then it empties the rewards buffer. This method should be called at the end of each episode.
        '''
        for key in self.rewards_dict:
            self.avg_rewards_dict[key].append(np.mean(self.rewards_dict[key]))
            self.rewards_dict[key] = []

        if plot:
            for key in self.avg_rewards_dict:
                avg_rewards = self.avg_rewards_dict[key]
                plt.subplot(2,1,1)
                plt.title('Average Reward / Episode')
                plt.plot(range(len(avg_rewards)), avg_rewards, label=key)
            for key in self.cumulative_rewards_dict:
                cumul_rewards = self.cumulative_rewards_dict[key]
                plt.subplot(2, 1, 2)
                plt.title('Cumulative rewards')
                plt.plot(range(len(cumul_rewards)), cumul_rewards, label=key)

            plt.tight_layout()
            plt.legend()
            plt.draw()
            plt.pause(0.001)
            plt.savefig('result.PNG')
            plt.clf()

    def get_learn_curve(self):
        return self.avg_rewards_dict