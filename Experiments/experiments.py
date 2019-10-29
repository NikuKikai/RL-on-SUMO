from Environment.sumoenv import SumoEnv
from Environment.traffic_system_manager import TrafficSystemManager
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
        self.env = SumoEnv(self.args, capture_path=self.res_dir_path)

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

            manager.dump_data_on_episode_end(self.res_dir_path)
            self.env.close()


def main_experiment(arguments):
    list_experiments = [
        {'path': './logs/compare/density__wt_max/', 'stype': 'density', 'rtype': 'wt_max'},
        {'path': './logs/compare/position__wt_max/', 'stype': 'position', 'rtype': 'wt_max'},
        {'path': './logs/compare/mean_speed__wt_max/', 'stype': 'mean_speed', 'rtype': 'wt_max'},
        {'path': './logs/compare/queue__wt_max/', 'stype': 'queue', 'rtype': 'wt_max'},
        {'path': './logs/compare/vehicle_types_emergency__wt_max/', 'stype': 'vehicle_types_emergency',
         'rtype': 'wt_max'},
        {'path': './logs/compare/density_and_speed__wt_max/', 'stype': 'density_and_speed', 'rtype': 'wt_max'},
        {'path': './logs/compare/density_speed_emergency__wt_max/', 'stype': 'density_speed_emergency',
         'rtype': 'wt_max'},
        {'path': './logs/compare/density_queue__wt_max/', 'stype': 'density_queue', 'rtype': 'wt_max'},
        {'path': './logs/compare/density_queue_mean_speed__wt_max/', 'stype': 'density_queue_mean_speed',
         'rtype': 'wt_max'},
        {'path': './logs/compare/density__wt_total_acc_relative/', 'stype': 'density', 'rtype': 'wt_total_acc_relative'},
        {'path': './logs/compare/position__wt_total_acc_relative/', 'stype': 'position',
         'rtype': 'wt_total_acc_relative'},
        {'path': './logs/compare/mean_speed__wt_total_acc_relative/', 'stype': 'mean_speed',
         'rtype': 'wt_total_acc_relative'},
        {'path': './logs/compare/queue__wt_total_acc_relative/', 'stype': 'queue', 'rtype': 'wt_total_acc_relative'},
        {'path': './logs/compare/vehicle_types_emergency__wt_total_acc_relative/', 'stype': 'vehicle_types_emergency',
         'rtype': 'wt_total_acc_relative'},
        {'path': './logs/compare/density_and_speed__wt_total_acc_relative/', 'stype': 'density_and_speed',
         'rtype': 'wt_total_acc_relative'},
        {'path': './logs/compare/density_speed_emergency__wt_total_acc_relative/', 'stype': 'density_speed_emergency',
         'rtype': 'wt_total_acc_relative'},
        {'path': './logs/compare/density_queue__wt_total_acc_relative/', 'stype': 'density_queue',
         'rtype': 'wt_total_acc_relative'},
        {'path': './logs/compare/density_queue_mean_speed__wt_total_acc_relative/', 'stype': 'density_queue_mean_speed',
         'rtype': 'wt_total_acc_relative'},
        {'path': './logs/compare/vehicle_types_emergency__wt_vehicle_class/', 'stype': 'vehicle_types_emergency',
         'rtype': 'wt_vehicle_class'},
        {'path': './logs/compare/density_speed_emergency__wt_max/', 'stype': 'density_speed_emergency',
         'rtype': 'wt_max'},
    ]
    for exp_dict in list_experiments:
        arguments.experiment_res_path = exp_dict['path']
        arguments.state_type = exp_dict['stype']
        arguments.reward_type = exp_dict['rtype']
        exp = Experiment(arguments)
        exp.train_default()
    return