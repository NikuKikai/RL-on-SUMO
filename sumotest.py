import numpy as np
import sumoenv as se
import matplotlib.pyplot as plt
from rl_agents import Fixed_Q_Targets_Agent
from rl_agents import DQN_Agent
from rl_agents import Cyclic_Agent
from rl_args import fixed_q_targets_args
from rl_args import dqn_args

args = fixed_q_targets_args()
#args = dqn_args()
env_train = se.SumoEnv(args, gui_f=False)
env_test = se.SumoEnv(args, gui_f=True)

agent = Fixed_Q_Targets_Agent(84, 4, args)

#agent = DQN_Agent(84, 4, args, device='cuda')

plt.ion()

cumm_rewards = []
for _ in range(args.episodes):
    state = env_train.reset(heatup=args.sim_heatup)
    done = False
    rewards_list = []
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, rewards = env_train.do_step(action)
        rewards_list.append(reward)
        agent.add_to_memory(state, action, next_state, reward) # TODO: missing done
        agent.optimize_model() # try to optimize if enough samples in memory.
        state = next_state
    cumm_rewards.append(np.mean(rewards_list))
    plt.plot(range(len(cumm_rewards)), cumm_rewards)
    plt.draw()
    plt.pause(0.001)
    plt.clf()

    env_train.close()
    agent.dump_rewards()

state = env_test.reset()
done = False
while not done:
    action = agent.select_action(state)
    next_state, reward, done, rewards = env_test.step_d(action)
    state = next_state
env_test.close()


