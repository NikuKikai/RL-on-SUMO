import os
import sys
import traci
import numpy as np


class SumoEnv:
    place_len = 7.5
    place_offset = 8.50
    lane_len = 10
    lane_ids = ['-gneE0_0', '-gneE0_1', '-gneE1_0', '-gneE1_1', '-gneE2_0', '-gneE2_1', '-gneE3_0', '-gneE3_1']

    def __init__(self, label='default', gui_f=False):
        self.label = label
        self.wt_last = 0.
        self.ncars = 0

        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        exe = 'sumo-gui.exe' if gui_f else 'sumo.exe'
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', exe)
        self.sumoCmd = [sumoBinary, '-c', 'intersection.sumocfg']

        return

    def get_state_d(self):
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
                state[int(ilane * self.lane_len + ipos)] += 1. - pos + ipos
                state[int(ilane * self.lane_len + ipos + 1)] += pos - ipos
            state[self.lane_len * 8:self.lane_len * 8+4] = np.eye(4)[traci.trafficlight.getPhase('gneJ00')]
        return state

    def step_d(self, action):
        done = False
        # traci.switch(self.label)

        action = np.squeeze(action)
        traci.trafficlight.setPhase('gneJ00', action)

        traci.simulationStep()
        traci.simulationStep()

        self.ncars += traci.simulation.getDepartedNumber()

        state = self.get_state_d()

        wt = 0
        for ilane in range(0, 8):
            lane_id = self.lane_ids[ilane]
            wt += traci.lane.getWaitingTime(lane_id)
        reward = - (wt - self.wt_last)*0.004

        if self.ncars > 250:
            done = True

        return state, reward, done, np.array([[reward]])

    def reset(self):
        self.wt_last = 0.
        self.ncars = 0
        traci.start(self.sumoCmd, label=self.label)
        traci.trafficlight.setProgram('gneJ00', '0')
        traci.simulationStep()
        return self.get_state_d()

    def close(self):
        traci.close()
