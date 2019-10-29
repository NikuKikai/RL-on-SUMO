import argparse
from Agents.rl_args import *


def load_agents_args(agent, network_type):
    if agent == 'fixed_q_targets':
        if network_type == 'single':
            return fixed_q_targets_single_intersection_args()
        elif network_type == 'double':
            return fixed_q_targets_double_intersection_args()
        else:
            print("Unsupported network")
            exit()
    elif agent == 'double_dqn':
        if network_type == 'single':
            return double_dqn_single_intersection_args()
        elif network_type == 'double':
            return double_dqn_double_intersection_args()
        else:
            print("Unsupported network")
            exit()
    else:
        print("Unsupported agent")
        exit()


def process_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Reinforcement learning project - Traffic lights control" \
                                                 "\nTechnion - Israel institute of technology\n" \
                                                 "Authors:\n" \
                                                 "\tAlexey Tusov, tusovalexey[at]gmail.com\n" \
                                                 "\tPavel Rastopchin, pavelr[at]gmail.com\n" \
                                                 "\tAmeen Ali, ameen.ali[at]gmail.com",
                                     epilog="example usage:" \
                                            "python ./main.py")
    parser.add_argument("-sc", "--sumo-cfg", type=str, default='./networks/double/Israel_zipping/network.sumocfg', dest='cfg',
                        help='path to desired simulation configuration file, default: ./networks/double/Israel_zipping/network.sumocfg')
    parser.add_argument("-a", "--agent", type=str, default='fixed_q_targets', dest='agent',
                        help='RL agent to use, supported agents: [fixed_q_targets, double_dqn] , default: fixed_q_targets')
    args, remaining = parser.parse_known_args()
    agent = args.agent
    network_type = args.cfg.split('/')[2] # parsing network type - single / double
    agents_args = load_agents_args(agent, network_type)
    parser.add_argument("-e", "--episodes", type=int, default=agents_args.episodes, dest='episodes',
                        help='Number of episodes for simulation, default: agent\'s default value')
    parser.add_argument("-bs", "--batch-size", type=int, default=agents_args.batch_size, dest='batch_size',
                        help='Batch size, default: agent\'s default value')
    parser.add_argument("-eps", "--epsilon-decay", type=float, default=agents_args.eps_decay, dest='epsilon_decay',
                        help='Epsilon decay rate, default: agent\'s default value')
    parser.add_argument("-st", "--state-type", type=str, default=agents_args.state_type, dest='state_type',
                        help='State representation type, supported types: [density, position, '
                             'density_and_speed, density_speed_emergency, density_speed_bus,'
                             ' density_speed_bus_emergency], default: agent\'s default type')
    parser.add_argument("-rt", "--reward-type", type=str, default=agents_args.reward_type, dest='reward_type',
                        help='Reward calculation type, supported types: [ wt_sum_absolute, wt_avg_absolute,'
                             'wt_sum_relative, wt_max, accumulated_wt_max, wt_squares_sum,'
                             'wt_parametric, wt_vehicle_class], default: agent\'s default type')
    parser.add_argument("-nn", "--nn-layers", nargs='+', type=int, default=agents_args.layers, dest='nn_layers',
                        help='NN layers, input example: 10 20 40 would be translated to [10, 20, 40],'
                             ' default: agent\'s default type')
    parser.add_argument("-lp", "--log-path", type=str, default=agents_args.logs_root_path, dest='log_path',
                        help='Root dir for result logs, default: agent\'s default type')
    parser.add_argument("-gc", "--grad-clip", type=int, default=agents_args.grad_clip, dest='grad_clip',
                        help='Grad clip value, default: agent\'s default value')
    parser.add_argument("-ta", "--target-update", type=int, default=agents_args.target_update, dest='target_update',
                        help='Each how many steps to update target nn, default: agent\'s default value')
    parser.add_argument("-ce", "--capture-each", type=int, default=agents_args.capture_each, dest='capture_each',
                        help='Each how many episodes to capture, default: agent\'s default value')
    parser.add_argument("-rs", "--replay_size", type=int, default=agents_args.replay_size, dest='replay_size',
                        help='Capacity of replay buffer, default: agent\'s default value')
    parser.add_argument("-sl", "--sim-length", type=int, default=agents_args.sim_max_steps, dest='sim_length',
                        help='Simulation length in steps, default: agent\'s default value')
    parser.add_argument("-gm", "--gui-mode", default='disable', type=str, dest='gui',
                        help='GUI mode: [diasble, enable, capture], default: disable')
    args, remaining = parser.parse_known_args()
    # Update agents arguments based on parsed users arguments
    agents_args.episodes = args.episodes
    agents_args.batch_size = args.batch_size
    agents_args.eps_decay = args.epsilon_decay
    agents_args.state_type = args.state_type
    agents_args.reward_type = args.reward_type
    agents_args.layers = args.nn_layers
    agents_args.logs_root_path = args.log_path
    agents_args.grad_clip = args.grad_clip
    agents_args.target_update = args.target_update
    agents_args.capture_each = args.capture_each
    agents_args.replay_size = args.replay_size
    agents_args.sim_max_steps = args.sim_length
    agents_args.sim_file = args.cfg
    if agents_args.capture_each > 0:
        agents_args.gui = 'capture'
    else:
        agents_args.gui = args.gui
    # Optional future params
    #parser.add_argument("-load", default=False, action='store_true', dest='load',
    #                    help='load saved weights and net data from training, default: False')
    #parser.add_argument("-save", default=False, action='store_true', dest='save',
    #                    help='save network weights and net data after training for future use, default: False')
    parser.add_argument("-m", "--mode", type=str, default='experiment', dest='mode',
                        help='Execution mode: basic or experiment, default: experiment')
    args = parser.parse_args()
    # Update logs dir depend on execution mode
    agents_args.experiment_res_path = agents_args.logs_root_path + args.mode + '/'
    return args.mode, agents_args
