import numpy as np
import agent as ag
import sumoenv as se
import matplotlib.pyplot as plt
from rl_agents import Fixed_Q_Targets_Agent
from rl_agents import DQN_Agent
from rl_agents import Cyclic_Agent
from rl_args import fixed_q_targets_args
from rl_args import dqn_args

#args = fixed_q_targets_args()
args = dqn_args()
env_train = se.SumoEnv(args, gui_f=False)
env_test = se.SumoEnv(args, gui_f=True)

#agent2 = Fixed_Q_Targets_Agent(84, 4, args, device='cuda')

agent2 = DQN_Agent(84, 4, args, device='cuda')

#cyclic_agent = Cyclic_Agent(4)
# state = env_test.reset()
# done = False
# while not done:
#     action = cyclic_agent.select_action(state)
#     print(action)
#     next_state, reward, done, rewards = env_test.step_d(action)
#     cyclic_agent.add_to_memory(state, action, next_state, reward)
#     state = next_state
# env_test.close()
# cyclic_agent.dump_rewards()

for _ in range(args.episodes):
    state = env_train.reset(heatup=args.sim_heatup)
    done = False
    rewards_list = []
    while not done:
        action = agent2.select_action(state)
        next_state, reward, done, rewards = env_train.step_d(action)
        agent2.add_to_memory(state, action, next_state, reward)
        agent2.optimize_model() # try to optimize if enough samples in memory.
        state = next_state
    env_train.close()
    agent2.dump_rewards()

state = env_test.reset()
done = False
while not done:
    action = agent2.select_action(state)
    next_state, reward, done, rewards = env_test.step_d(action)
    state = next_state
env_test.close()


