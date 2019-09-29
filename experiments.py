import sumoenv as se
from rl_args import fixed_q_targets_israel_double_args
from rl_args import fixed_q_targets_args
from traffic_system_manager import TrafficSystemManager
import matplotlib.pyplot as plt

class DoubleIsraelNumOfEpisodes():
    '''
    this experiment is built to determine the issue related to double
    Israel intersection and noisy results if too many or not enough episodes.
    '''
    def __init__(self):
        self.args = fixed_q_targets_israel_double_args()

        self.env = se.SumoEnv(self.args, path_to_sim_file='simulations\\israel_double_intersection.sumocfg', gui=False)

    def train_default(self):
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
            manager.dump_learn_curve(plot=True)
            self.env.close()

    def train_with_different_num_of_episodes(self):
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