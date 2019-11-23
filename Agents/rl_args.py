class default_args():
    '''
    In this class we define the default arguments.
    '''
    def __init__(self):
        # Exploration
        self.eps_start = 0.99
        self.eps_end = 0.05

        # RL param
        self.discount = 0.999

        # Simulation params
        self.sim_heatup = 20  # simulation run this amount of steps after reset.
        self.sim_max_steps = 3000 # 3000 # simulation ends after this amount of steps.
        self.sim_steps_per_do_step = 6  # simulation steps performed between two actions.
        self.yellow_phase_duration = 3  # simulation steps performed between two actions.
        self.gui = False #'enable'
        self.state_type = 'density_and_speed'  # 'density', 'position', 'density_and_speed'
        self.reward_type = 'wt_sum_absolute'  # 'wt_sum_absolute', 'wt_avg_absolute', 'wt_sum_relative', 'wt_max', 'accumulated_wt_max', 'wt_squares_sum'

        # Replay params
        self.replay_size = 50000  # capacity of replay.

        # Deep q networks params
        self.batch_size = 512 # batch size feed into deep q net.
        self.optimizer = 'rmsprop' # Adam works bad.
        self.layers = [128, 64, 16] # deep net layers.

        # Results path
        self.logs_root_path = './logs/'
        self.experiment_res_path = './logs/experiments/'

        # Capture args:
        self.capture_each = -1 # Do capture each 20 episodes.

class single_intersection_args(default_args):
    def __init__(self):
        super().__init__()
        # Simulation params
        self.sim_file = './networks/single/Israel/network.sumocfg'
        # RL param
        self.episodes = 100  # 200 should be enough
        self.eps_decay = 100  # exploration decay rate. bigger is slower. Don't set it above 10000 - it too slow.

class double_intersection_args(default_args):
    def __init__(self):
        super().__init__()
        # Simulation params
        self.sim_file = './networks/double/Israel_zipping_yellow_light/network.sumocfg' # './networks/double/Israel/network.sumocfg'
        # RL param
        self.episodes = 150 # 300
        self.eps_decay = 100  # exploration decay rate. bigger is slower. Don't set it above 10000 - it too slow.


'''
### ARGUMENTS FOR DIFFERENT AGENTS AND DIFFERENT ENVIRONMENTS ###
'''
class cyclic_agent_double_intersection_args(double_intersection_args):
    def __init__(self):
        super().__init__()
        self.rl_algo = 'cyclic'
        self.episodes = 150


class random_agent_double_intersection_args(double_intersection_args):
    def __init__(self):
        super().__init__()
        self.rl_algo = 'random'
        self.episodes = 150


class fixed_q_targets_single_intersection_args(single_intersection_args):
    '''
    After experimenting we came up with those params.
    '''
    def __init__(self):
        super().__init__()
        # RL params
        self.target_update = 50 # target net updated each k steps
        self.rl_algo = 'fixed_q_targets'
        # DL args
        self.grad_clip = 1


class double_dqn_single_intersection_args(single_intersection_args):
    '''
    No optimal args found
    '''
    def __init__(self):
        super().__init__()
        self.target_update = 0.001  # polyak averaging
        self.rl_algo = 'double_dqn'
        # DL args
        self.grad_clip = 1


class fixed_q_targets_double_intersection_args(double_intersection_args):
    '''
    No optimal args found.
    '''
    def __init__(self):
        super().__init__()
        # RL params
        self.target_update = 50 # target net updated each k steps
        self.rl_algo = 'fixed_q_targets'
        # DL args
        self.grad_clip = 1


class double_dqn_double_intersection_args(double_intersection_args):
    '''
    After experimenting we came up with those params.
    '''
    def __init__(self):
        super().__init__()
        # RL params
        self.target_update = 0.001  # polyak averaging
        self.rl_algo = 'double_dqn'
        # DL args
        self.grad_clip = 400
