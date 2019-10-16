from sumoenv import SumoEnv
from traffic_system_manager import TrafficSystemManager
import matplotlib.pyplot as plt
import os
from datetime import datetime

class Experiment():
    def __init__(self, args, show_training_curve=True):
        self.args = args
        self.show_training_curve = show_training_curve

        # Create folder for experiments results
        root_path = args.experiment_res_path
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        timestamp = str(datetime.now()).replace(':', '_').replace('-', '_').replace('.', '_').replace(' ', '_')
        sim_name = os.path.split(args.sim_file)[1].split('.')[0]
        folder_name = timestamp + '_' + args.rl_algo + '_' + sim_name
        self.res_dir_path = os.path.join(root_path, folder_name)
        os.makedirs(self.res_dir_path)

        # Create env.
        self.env = SumoEnv(self.args, path_to_sim_file=args.sim_file,
                           always_gui=False, capture_each=args.capture_each,
                           capture_path=self.res_dir_path)

        # Save args
        self.args_file_path = os.path.join(self.res_dir_path, 'args.txt')
        with open(self.args_file_path, 'a') as file:
            for key in vars(args):
                file.write(str(key)+': '+str(vars(args)[key])+'\n')

    def train_default(self):
        manager = TrafficSystemManager(self.env.get_dimensions(), self.args)
        for epi in range(self.args.episodes):
            print("### Starting Episode: ", epi, ' ###')
            state = self.env.reset(heatup=self.args.sim_heatup)
            done = False
            while not done:
                action = manager.select_action(state)
                next_state, reward, done = self.env.do_step(action)
                manager.add_to_memory(state, action, next_state, reward)  # TODO: missing done
                manager.teach_agents()  # try to optimize if enough samples in memory.
                state = next_state

            manager.dump_data_on_episode_end(self.res_dir_path, plot=self.show_training_curve)
            self.env.close()

    def train_with_different_reward_types(self):
        '''
        This experiment checks all reward types.
        '''
        learning_curves_list_first_gneJ = []
        learning_curves_list_second_gneJ = []
        for reward_type in ['wt_sum_absolute', 'wt_sum_relative', 'wt_max', 'accumulated_wt_max', 'wt_squares_sum']:
            print("### Reward type: ", reward_type,' ###')
            self.args.reward_type = reward_type
            self.env = SumoEnv(self.args, path_to_sim_file='simulations\\israel_double_intersection.sumocfg',
                                  gui=False)
            manager = TrafficSystemManager(self.env.get_dimensions(), self.args)
            for _ in range(self.args.episodes):
                state = self.env.reset(heatup=self.args.sim_heatup)
                done = False
                while not done:
                    action = manager.select_action(state)
                    next_state, reward, done = self.env.do_step(action)
                    manager.add_to_memory(state, action, next_state, reward)  # TODO: missing done
                    manager.teach_agents()  # try to optimize if enough samples in memory.
                    state = next_state
                manager.dump_learn_curve(plot=False)
                self.env.close()
            learning_curves_list_first_gneJ.append(manager.get_learn_curve()['gneJ0'])
            learning_curves_list_second_gneJ.append(manager.get_learn_curve()['gneJ6'])

        # plot results
        for res in learning_curves_list_first_gneJ:
            plt.subplot(1,2,1)
            plt.title('gneJ0')
            plt.plot(range(len(res)), res)
        for res in learning_curves_list_second_gneJ:
            plt.subplot(1,2,2)
            plt.title('gneJ6')
            plt.plot(range(len(res)), res)
        plt.show()

    def train_with_different_num_of_steps(self):
        '''
        This experiment is built to determine the optimal number of steps of simulation
        that sould be performed each do_step() call.
        '''
        num_of_sim_steps_list = [1, 2, 3, 4]
        learning_curves_list_first_gneJ = []
        learning_curves_list_second_gneJ = []
        for num_of_sim_steps in num_of_sim_steps_list:
            print('### Num of steps: ', num_of_sim_steps, ' ###')
            self.args.sim_steps = num_of_sim_steps

            # approximately, number of episodes needed is x10 of sim_steps.
            self.args.episodes = num_of_sim_steps * 30
            self.env = SumoEnv(self.args, path_to_sim_file='simulations\\israel_double_intersection.sumocfg',
                                  gui=False)
            manager = TrafficSystemManager(self.env.get_dimensions(), self.args)
            for _ in range(self.args.episodes):
                state = self.env.reset(heatup=self.args.sim_heatup)
                done = False
                while not done:
                    action = manager.select_action(state)
                    next_state, reward, done = self.env.do_step(action)
                    manager.add_to_memory(state, action, next_state, reward)  # TODO: missing done
                    manager.teach_agents()  # try to optimize if enough samples in memory.
                    state = next_state
                manager.dump_learn_curve(plot=False)
                self.env.close()
            learning_curves_list_first_gneJ.append(manager.get_learn_curve()['gneJ0'])
            learning_curves_list_second_gneJ.append(manager.get_learn_curve()['gneJ6'])

        # plot results
        plot_cols_n = 2 # two agents.
        plot_rows_n = len(num_of_sim_steps_list)

        for idx, res in enumerate(learning_curves_list_first_gneJ):
            plt.subplot(plot_rows_n,plot_cols_n,plot_cols_n*idx+1)
            if idx==0:
                plt.title('gneJ0')
            lbl = 'sim steps ' + str(num_of_sim_steps_list[idx])
            plt.plot(range(len(res)), res, label=lbl)
            plt.legend()
        for idx, res in enumerate(learning_curves_list_second_gneJ):
            plt.subplot(plot_rows_n,plot_cols_n,plot_cols_n*idx+2)
            if idx==0:
                plt.title('gneJ6')
            lbl = 'sim steps ' + str(num_of_sim_steps_list[idx])
            plt.plot(range(len(res)), res, label=lbl)
            plt.legend()

        plt.tight_layout()
        plt.show()

    def train_with_different_num_of_episodes(self):
        '''
        this experiment is built to determine the issue related to double
        Israel intersection and noisy results if too many or not enough episodes.
        '''
        num_of_episodes_list = [60, 80, 100, 120, 140, 160, 180, 200]
        learning_curves_list_first_gneJ = []
        learning_curves_list_second_gneJ = []
        for num_of_episodes in num_of_episodes_list:
            manager = TrafficSystemManager(self.env.get_dimensions(), self.args)
            for _ in range(num_of_episodes):
                state = self.env.reset(heatup=self.args.sim_heatup)
                done = False
                while not done:
                    action = manager.select_action(state)
                    next_state, reward, done = self.env.do_step(action)
                    manager.add_to_memory(state, action, next_state, reward)  # TODO: missing done
                    manager.teach_agents()  # try to optimize if enough samples in memory.
                    state = next_state
                manager.dump_learn_curve(plot=False)
                self.env.close()
            learning_curves_list_first_gneJ.append(manager.get_learn_curve()['gneJ0'])
            learning_curves_list_second_gneJ.append(manager.get_learn_curve()['gneJ6'])

        # plot results
        for res in learning_curves_list_first_gneJ:
            plt.subplot(1,2,1)
            plt.title('gneJ0')
            plt.plot(range(len(res)), res)
        for res in learning_curves_list_second_gneJ:
            plt.subplot(1,2,2)
            plt.title('gneJ6')
            plt.plot(range(len(res)), res)
        plt.show()