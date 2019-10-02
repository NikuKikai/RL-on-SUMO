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
        self.sim_steps = 3  # simulation steps performed between two actions.
        self.state_type = 'density_and_speed'  # 'density', 'position', 'density_and_speed'
        self.reward_type = 'wt_sum_absolute'  # 'wt_sum_absolute', 'wt_avg_absolute', 'wt_sum_relative', 'wt_max', 'accumulated_wt_max', 'wt_squares_sum'

        # Replay params
        self.replay_size = 50000  # capacity of replay.

        # Deep q networks params
        self.batch_size = 512 # batch size feed into deep q net.
        self.optimizer = 'rmsprop' # Adam works bad.
        self.layers = [128, 64, 16] # deep net layers.


class single_intersection_args(default_args):
    def __init__(self):
        super().__init__()
        # Simulation params
        self.sim_file = 'simulations\\israel_single_intersection.sumocfg'
        # RL param
        self.episodes = 200  # 200 should be enough
        self.eps_decay = 100  # exploration decay rate. bigger is slower. Don't set it above 10000 - it too slow.

class double_intersection_args(default_args):
    def __init__(self):
        super().__init__()
        # Simulation params
        self.sim_file = 'simulations\\israel_double_intersection.sumocfg'
        # RL param
        self.episodes = 300 # 300
        self.eps_decay = 100  # exploration decay rate. bigger is slower. Don't set it above 10000 - it too slow.


'''
### ARGUMENTS FOR DIFFERENT AGENTS AND DIFFERENT ENVIRONMENTS ###
'''

class fixed_q_targets_single_intersection_args(single_intersection_args):
    '''
    After experimenting I (pavel) came up with those params.
    It should be compared with other agents.
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
    No optimal args found
    '''
    def __init__(self):
        super().__init__()
        # RL params
        self.target_update = 0.001  # polyak averaging
        self.rl_algo = 'double_dqn'
        # DL args
        self.grad_clip = None
