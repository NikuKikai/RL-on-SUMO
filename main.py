from Experiments.experiments import Experiment
from Agents.rl_args import *
from utils.process_args import process_arguments
from Environment.sumoenv import SumoEnv
from Environment.traffic_system_manager import TrafficSystemManager
from datetime import datetime
import os

### disable some plt warnings.
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
###



def main_experiment(arguments):
    list_experiments = [
        {'path': '/logs/compare/density__wt_max/', 'stype': 'density', 'rtype': 'wt_max'},
        {'path': '/logs/compare/position__wt_max/', 'stype': 'position', 'rtype': 'wt_max'},
        {'path': '/logs/compare/mean_speed__wt_max/', 'stype': 'mean_speed', 'rtype': 'wt_max'},
        {'path': '/logs/compare/queue__wt_max/', 'stype': 'queue', 'rtype': 'wt_max'},
        {'path': '/logs/compare/vehicle_types_emergency__wt_max/', 'stype': 'vehicle_types_emergency',
         'rtype': 'wt_max'},
        {'path': '/logs/compare/density_and_speed__wt_max/', 'stype': 'density_and_speed', 'rtype': 'wt_max'},
        {'path': '/logs/compare/density_speed_emergency__wt_max/', 'stype': 'density_speed_emergency',
         'rtype': 'wt_max'},
        {'path': '/logs/compare/density_queue__wt_max/', 'stype': 'density_queue', 'rtype': 'wt_max'},
        {'path': '/logs/compare/density_queue_mean_speed__wt_max/', 'stype': 'density_queue_mean_speed',
         'rtype': 'wt_max'},
        {'path': '/logs/compare/density__wt_total_acc_relative/', 'stype': 'density', 'rtype': 'wt_total_acc_relative'},
        {'path': '/logs/compare/position__wt_total_acc_relative/', 'stype': 'position',
         'rtype': 'wt_total_acc_relative'},
        {'path': '/logs/compare/mean_speed__wt_total_acc_relative/', 'stype': 'mean_speed',
         'rtype': 'wt_total_acc_relative'},
        {'path': '/logs/compare/queue__wt_total_acc_relative/', 'stype': 'queue', 'rtype': 'wt_total_acc_relative'},
        {'path': '/logs/compare/vehicle_types_emergency__wt_total_acc_relative/', 'stype': 'vehicle_types_emergency',
         'rtype': 'wt_total_acc_relative'},
        {'path': '/logs/compare/density_and_speed__wt_total_acc_relative/', 'stype': 'density_and_speed',
         'rtype': 'wt_total_acc_relative'},
        {'path': '/logs/compare/density_speed_emergency__wt_total_acc_relative/', 'stype': 'density_speed_emergency',
         'rtype': 'wt_total_acc_relative'},
        {'path': '/logs/compare/density_queue__wt_total_acc_relative/', 'stype': 'density_queue',
         'rtype': 'wt_total_acc_relative'},
        {'path': '/logs/compare/density_queue_mean_speed__wt_total_acc_relative/', 'stype': 'density_queue_mean_speed',
         'rtype': 'wt_total_acc_relative'},
        {'path': '/logs/compare/vehicle_types_emergency__wt_vehicle_class/', 'stype': 'vehicle_types_emergency',
         'rtype': 'wt_vehicle_class'},
        {'path': '/logs/compare/density_speed_emergency__wt_max/', 'stype': 'density_speed_emergency',
         'rtype': 'wt_max'},
    ]
    for exp_dict in list_experiments:
        arguments.experiment_res_path = exp_dict['path']
        arguments.state_type = exp_dict['stype']
        arguments.reward_type = exp_dict['rtype']
        exp = Experiment(arguments)
        exp.train_default()
    return


def main(arguments):
    # Create folder for experiments results
    root_path = arguments.experiment_res_path
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    timestamp = str(datetime.now()).replace(':', '_').replace('-', '_').replace('.', '_').replace(' ', '_')
    sim_name = os.path.split(arguments.sim_file)[1].split('.')[0]
    folder_name = timestamp + '_' + arguments.rl_algo + '_' + sim_name
    res_dir_path = os.path.join(root_path, folder_name)
    os.makedirs(res_dir_path)

    # Create env.
    env = SumoEnv(arguments, capture_path=res_dir_path)

    # Save args
    args_file_path = os.path.join(res_dir_path, 'args.txt')
    with open(args_file_path, 'a') as file:
        for key in vars(arguments):
            file.write(str(key) + ': ' + str(vars(arguments)[key]) + '\n')

    manager = TrafficSystemManager(env.get_dimensions(), arguments)
    for epi in range(arguments.episodes):
        print("### Starting Episode: ", epi, ' ###')
        state = env.reset(heatup=arguments.sim_heatup)
        done = False
        while not done:
            action = manager.select_action(state)
            next_state, reward, done = env.do_step(action)
            manager.add_to_memory(state, action, next_state, reward)
            manager.teach_agents()  # try to optimize if enough samples in memory.
            state = next_state

        manager.dump_data_on_episode_end(res_dir_path)
        env.close()
    return


if __name__ == '__main__':
    mode, args = process_arguments()
    if mode == 'experiment':
        main_experiment(args)
    else:
        main(args)
