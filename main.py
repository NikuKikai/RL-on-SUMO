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


def main_experiment():
    # # Experiment 1
    experiment_args = fixed_q_targets_single_intersection_args()
    experiment_args.target_update = 50
    exp = Experiment(experiment_args)
    exp.train_default()
    experiment_args = fixed_q_targets_single_intersection_args()
    experiment_args.target_update = 0.001
    exp = Experiment(experiment_args)
    exp.train_default()

    # # Experiment 2
    experiment_args = fixed_q_targets_double_intersection_args()
    experiment_args.target_update = 50
    exp = Experiment(experiment_args)
    exp.train_default()
    experiment_args = fixed_q_targets_double_intersection_args()
    experiment_args.target_update = 0.001
    exp = Experiment(experiment_args)
    exp.train_default()

    # # Experiment 3
    experiment_args = double_dqn_single_intersection_args()
    experiment_args.target_update = 50
    exp = Experiment(experiment_args)
    exp.train_default()
    experiment_args = double_dqn_single_intersection_args()
    experiment_args.target_update = 0.001
    exp = Experiment(experiment_args)
    exp.train_default()

    # Experiment 4
    experiment_args = double_dqn_double_intersection_args()
    experiment_args.target_update = 50
    exp = Experiment(experiment_args)
    exp.train_default()
    experiment_args = double_dqn_double_intersection_args()
    args.grad_clip = True
    exp = Experiment(experiment_args)
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
        main_experiment()
    else:
        main(args)
