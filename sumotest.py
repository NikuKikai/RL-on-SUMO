import numpy as np
import agent as ag
import sumoenv as se

env_train = se.SumoEnv(gui_f=False)
env_test = se.SumoEnv(gui_f=True)
agent = ag.Agent()

EPS = 20

for ieps in range(EPS):
    for i in range(20):
        state = env_train.reset()
        done = False
        while not done:
            action = agent.policy(state)
            next_state, reward, done, rewards = env_train.step_d(action)

            agent.train(state, action, reward, 0.001, [1, 1, done, 1, 1])

            state = next_state
        env_train.close()

    state = env_test.reset()
    done = False
    while not done:
        action = agent.policy(state)
        next_state, reward, done, rewards = env_test.step_d(action)
        print(state)

        state = next_state
    env_test.close()

