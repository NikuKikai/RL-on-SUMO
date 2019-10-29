from Agents.rl_agents import Fixed_Q_Targets_Agent
from Agents.rl_agents import Double_DQN_Agent
from Agents.stupid_agents import Cyclic_Agent
from Agents.stupid_agents import Random_Agent
import numpy as np
import matplotlib.pyplot as plt
import os

class TrafficSystemManager:
    def __init__(self, dim_dict, rl_args, verbose=True):
        self.agents_dict = {}
        self.args = rl_args

        # Metrics logging:
        self.episode = 0
        self.avg_rewards_dict = {} # Average reward per episode
        self.rewards_dict = {} # Rewards of an episode
        self.cumulative_rewards_dict = {} # Cumulative rewards of entire experiment
        self.exploration_rate_dict = {} # for plotting exploration rate
        self.best_avg_reward = {} # for saving best model

        for intersection_name in dim_dict:
            input_state_size, num_of_actions = dim_dict[intersection_name]
            # TODO: enable other agents too..
            if rl_args.rl_algo == 'dqn':
                raise NotImplementedError
            elif rl_args.rl_algo == 'fixed_q_targets':
                self.agents_dict[intersection_name] = Fixed_Q_Targets_Agent(input_state_size, num_of_actions, rl_args)
            elif rl_args.rl_algo == 'double_dqn':
                self.agents_dict[intersection_name] = Double_DQN_Agent(input_state_size, num_of_actions, rl_args)
            elif rl_args.rl_algo == 'cyclic':
                self.agents_dict[intersection_name] = Cyclic_Agent(num_of_actions)
            elif rl_args.rl_algo == 'random':
                self.agents_dict[intersection_name] = Random_Agent(num_of_actions)
            else:
                raise NotImplementedError

            # Initialize all loggers.
            self.avg_rewards_dict[intersection_name] = []
            self.rewards_dict[intersection_name] = []
            self.cumulative_rewards_dict[intersection_name] = [0]
            self.exploration_rate_dict[intersection_name] = []
            self.best_avg_reward[intersection_name] = -np.inf

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

            # Add to replay
            agent_name = intersection_name  # for sake of explicity.
            self.agents_dict[agent_name].add_to_memory(state, action, next_state, reward)

            # Logging buffers
            self.rewards_dict[agent_name].append(reward)
            self.exploration_rate_dict[agent_name].append(self.agents_dict[agent_name].eps_threshold)
            last_cumulative_reward = self.cumulative_rewards_dict[agent_name][-1]
            self.cumulative_rewards_dict[agent_name].append(last_cumulative_reward + reward)

    def teach_agents(self):
        for agent_name in self.agents_dict:
            self.agents_dict[agent_name].optimize_model()

    def _save_best_models(self, res_path):
        '''
        This method iterates over all agents,
        saves checkpoints of policy nets if they got best reward.
        '''
        for key in self.rewards_dict:

            if not hasattr(self.agents_dict[key], 'policy_net'):
                # If agent doesn't have dqn, there is nothing to save.
                continue

            cur_avg_reward = np.mean(self.rewards_dict[key])
            if cur_avg_reward > self.best_avg_reward[key]:
                self.best_avg_reward[key] = cur_avg_reward
                # save checkpoint of policy model
                ckpt_folder = os.path.join(res_path, key+'_ckpt')
                if not os.path.exists(ckpt_folder):
                    os.makedirs(ckpt_folder)
                with open(os.path.join(ckpt_folder, 'model_stats.txt'), 'w') as file:
                    file.writelines('best_avg_reward:' + str(cur_avg_reward) + '\n')
                    file.writelines('episode:' + str(self.episode) + '\n')
                    file.writelines(str(self.agents_dict[key].policy_net))
                self.agents_dict[key].save_ckpt(ckpt_folder)

    def dump_data_on_episode_end(self, res_path, plot=False):
        '''
        This method should be called at the end of each episode. This method:
        1) Saves the best checkpoint.
        2) Saves the average reward of the performed steps up to this point.
        Then it empties the rewards buffer.
        3) Plot and save learning curve.
        '''
        # Save checkpoints
        self._save_best_models(res_path)

        # Update episode counter
        self.episode += 1

        for key in self.rewards_dict:
            self.avg_rewards_dict[key].append(np.mean(self.rewards_dict[key]))
            self.rewards_dict[key] = []

        for key in self.avg_rewards_dict.keys():
            avg_rewards = self.avg_rewards_dict[key]
            plt.subplot(3,1,1)
            plt.title('Average Reward')
            plt.plot(range(len(avg_rewards)), [0]*len(avg_rewards), linestyle=':', color='r')
            plt.plot(range(len(avg_rewards)), avg_rewards, label=key)
            plt.ylabel(self.args.reward_type)
            plt.xlabel('Episodes')
            plt.legend()
        for key in self.cumulative_rewards_dict.keys():
            cumul_rewards = self.cumulative_rewards_dict[key]
            plt.subplot(3, 1, 2)
            plt.title('Cumulative rewards')
            plt.plot(range(len(cumul_rewards)), cumul_rewards, label=key)
            plt.xlabel('TSM steps')
            plt.legend()
        for key in self.exploration_rate_dict.keys():
            exp_rate = self.exploration_rate_dict[key]
            plt.subplot(3, 1, 3)
            plt.title('Exploration Rate. Eps decay = ' + str(self.args.eps_decay))
            plt.plot(range(len(exp_rate)), exp_rate, label=key)
            plt.xlabel('TSM steps')
            plt.legend()

        fig = plt.gcf()
        fig.set_size_inches(9, 9)

        plt.savefig(os.path.join(res_path, 'learn_curve.PNG'))

        if plot:
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
        plt.clf()


    def get_learn_curve(self):
        return self.avg_rewards_dict
