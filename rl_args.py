
class fixed_q_targets_args():
    '''
    After experimenting I (pavel) came up with those params.
    It should be compared with other agents.
    '''
    # Exploration / Exploitation params.
    eps_start = 0.99
    eps_end = 0.05
    eps_decay = 100  # exploration decay rate. bigger is slower. Don't set it above 10000 - it too slow.

    # RL params
    episodes = 100 # 100 should be enough
    target_update = 50 # target net updated each k steps
    discount = 0.999 # dsicount of future reward.

    # Replay params
    replay_size = 5000 # capacity of replay.

    # Deep q networks params
    batch_size = 512 # batch size feed into deep q net.
    optimizer = 'rmsprop' # Adam works bad.

    # Simulation params
    sim_heatup = 50 # simulation run this amount of steps after reset.
    sim_max_steps = 1000 # simulation ends after this ammount of steps.
    sim_steps = 10 # simulation steps performed between two actions.
    state_type = 'density'

    # Logging params
    output_file_name = 'fixed_q_targets_output.txt'

class fixed_q_targets_israel_double_args():
    '''
    After experimenting I (pavel) came up with those params.
    It should be compared with other agents.
    '''
    # Exploration / Exploitation params.
    eps_start = 0.99
    eps_end = 0.05
    eps_decay = 1000  # exploration decay rate. bigger is slower. Don't set it above 10000 - it too slow.

    # RL params
    episodes = 50 # 100 should be enough
    target_update = 50 # target net updated each k steps
    discount = 0.999 # dsicount of future reward.

    # Replay params
    replay_size = 5000 # capacity of replay.

    # Deep q networks params
    batch_size = 512 # batch size feed into deep q net.
    optimizer = 'rmsprop' # Adam works bad.

    # Simulation params
    sim_heatup = 50 # simulation run this amount of steps after reset.
    sim_max_steps = 3000 # simulation ends after this amount of steps.
    sim_steps = 10 # simulation steps performed between two actions.
    state_type = 'density'

    # Logging params
    output_file_name = 'fixed_q_targets_israel_double_output.txt'

class dqn_args():
    '''
    No optimal performance found
    '''
    # Exploration / Exploitation params.
    eps_start = 0.99
    eps_end = 0.05
    eps_decay = 100  # exploration decay rate. bigger is slower. Don't set it above 10000 - it too slow.

    # RL params
    episodes = 400 # 100 should be enough
    target_update = None # target net updated each k steps
    discount = 0.999 # dsicount of future reward.

    # Replay params
    replay_size = 5000 # capacity of replay.

    # Deep q networks params
    batch_size = 512 # batch size feed into deep q net.
    optimizer = 'rmsprop' # Adam works bad.

    # Simulation params
    teleport_punishment = 50 # when vehicle waits too much it teleports. It should be punished.
    sim_heatup = 50 # simulation run this amount of steps after reset.
    sim_max_steps = 1500 # simulation ends after this ammount of steps.
    sim_steps = 10 # simulation steps performed between two actions.

    # Logging params
    output_file_name = 'dqn_output.txt'


class code_sanity_args():
    '''
    '''
    # Exploration / Exploitation params.
    eps_start = 0.99
    eps_end = 0.05
    eps_decay = 100  # exploration decay rate. bigger is slower. Don't set it above 10000 - it too slow.

    # RL params
    episodes = 5 # 100 should be enough
    target_update = 5 # target net updated each k steps
    discount = 0.8 # dsicount of future reward.

    # Replay params
    replay_size = 5000 # capacity of replay.

    # Deep q networks params
    batch_size = 2048 # batch size feed into deep q net.
    optimizer = 'rmsprop' # Adam works bad.

    # Simulation params
    teleport_punishment = 10 # when vehicle waits too much it teleports. It should be punished.
    sim_heatup = 10 # simulation run this amount of steps after reset.
    sim_max_steps = 500 # simulation ends after this ammount of steps.
    sim_steps = 5 # simulation steps performed between two actions.