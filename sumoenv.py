import os
import sys
import traci
import numpy as np
from rl_args import fixed_q_targets_args


class SumoEnv:
    place_len = 7.5
    place_offset = 8.50
    lane_len = 10
    lane_ids = ['-gneE0_0', '-gneE0_1', '-gneE1_0', '-gneE1_1', '-gneE2_0', '-gneE2_1', '-gneE3_0', '-gneE3_1']

    def __init__(self, args, label='default', max_steps=1000 ,sim_steps=10 ,gui_f=False):
        self.label = label
        self.wt_last = 0.
        self.ncars = 0
        self.max_steps = args.sim_max_steps

        # How much simulation steps perfromed in current episode.
        self.steps_done = 0
        # How much simulation steps perfromed each step_d call.
        self.sim_steps = args.sim_steps
        # How to punish for teleport
        self.teleport_punishment = args.teleport_punishment

        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        exe = 'sumo-gui.exe' if gui_f else 'sumo.exe'
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', exe)
        self.sumoCmd = [sumoBinary, '-c', 'simulations\\intersection.sumocfg']



        return

    def get_state_d(self):
        # Prepare empty state of fixed size.
        # The state contains 8 discrete lines + one hot vec of phase.
        state = np.zeros(self.lane_len * 8 + 4, dtype=np.float32)

        for ilane in range(0, 8):
            lane_id = self.lane_ids[ilane]
            ncars = traci.lane.getLastStepVehicleNumber(lane_id)
            cars = traci.lane.getLastStepVehicleIDs(lane_id)
            for icar in cars:
                xcar, ycar = traci.vehicle.getPosition(icar)
                if ilane < 2:
                    pos = (ycar - self.place_offset) / self.place_len
                elif ilane < 4:
                    pos = (xcar - self.place_offset) / self.place_len
                elif ilane < 6:
                    pos = (-ycar - self.place_offset) / self.place_len
                else:
                    pos = (-xcar - self.place_offset) / self.place_len
                if pos > self.lane_len - 1.:
                    continue
                pos = np.clip(pos, 0., self.lane_len - 1. - 1e-6)
                ipos = int(pos)

                # Each car exist at the same time in two cells, with some certainty.
                # In this way we can represent continuous position of a car on a discrete grid.
                state[int(ilane * self.lane_len + ipos)] += 1. - (pos - ipos)
                state[int(ilane * self.lane_len + ipos + 1)] += pos - ipos

            # Adding phase as one hot vector to state.
            state[self.lane_len * 8:self.lane_len * 8+4] = np.eye(4)[traci.trafficlight.getPhase('gneJ00')]
        return state

    def step_d(self, action):
        self.steps_done += self.sim_steps
        done = False
        # traci.switch(self.label)

        action = np.squeeze(action)
        traci.trafficlight.setPhase('gneJ00', action)

        # Perform multiple steps of simulation.
        for _ in range(self.sim_steps):
            traci.simulationStep()

        self.ncars += traci.simulation.getDepartedNumber()

        state = self.get_state_d()

        reward, _ = self.calc_reward()

        if self.ncars > 250 or self.steps_done >= self.max_steps:
            done = True
            self.steps_done = 0

        return state, reward, done, np.array([[reward]])

    def calc_reward(self):
        '''
        This function calculates reward. We can use rewards of two types:
        1) Absolute waiting time - just a sum of wait times at current time point.
        2) Relative wait time - An increase or decrease of wait time relatively to previous state.
        Both work well. Absolute reward is more stable and informative. See details in report.
        :return: absolute_reward, relative_reward
        '''
        wt = 0

        for ilane in range(0, 8):
            lane_id = self.lane_ids[ilane]
            wt += traci.lane.getWaitingTime(lane_id)

        relative_reward = - (wt - self.wt_last) * 50 #scale factor
        absolute_reward = - wt * 0.004 #scale factor

        self.wt_last = wt
        return absolute_reward, relative_reward

    def reset(self, heatup=50):
        self.wt_last = 0.
        self.ncars = 0
        traci.start(self.sumoCmd, label=self.label, )
        traci.trafficlight.setProgram('gneJ00', '0')
        for _ in range(heatup):
            traci.simulationStep()
        return self.get_state_d()

    def close(self):
        traci.close()
