
import sumoenv as se
from rl_args import fixed_q_targets_args
from traffic_system_manager import TrafficSystemManager

args = fixed_q_targets_args()

env = se.SumoEnv(args, path_to_sim_file='simulations\\israel_double_intersection.sumocfg', gui_f=False)
manager = TrafficSystemManager(env.get_road_structure(), args)

for _ in range(args.episodes):
    state = env.reset(heatup=args.sim_heatup)
    done = False
    while not done:
        action = manager.select_action(state)
        next_state, reward, done = env.do_step(action)
        manager.add_to_memory(state, action, next_state, reward) # TODO: missing done
        manager.teach_agents() # try to optimize if enough samples in memory.
        state = next_state
    env.close()
    manager.plot_learn_curve()