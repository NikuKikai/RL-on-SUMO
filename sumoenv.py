import os
import sys
import traci
import numpy as np
from math import ceil
from rl_args import fixed_q_targets_args


class SumoEnv:
    place_len = 7.5
    place_offset = 8.50

    lane_ids = ['-gneE0_0', '-gneE0_1', '-gneE1_0', '-gneE1_1', '-gneE2_0', '-gneE2_1', '-gneE3_0', '-gneE3_1']

    def __init__(self, args, path_to_sim_file='simulations\\intersection.sumocfg', label='default', max_steps=1000 ,sim_steps=10 ,gui_f=False):
        self.label = label
        self.wt_last = 0.
        self.ncars = 0
        self.max_steps = args.sim_max_steps
        self.path_to_sim_file = path_to_sim_file

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
        self.sumoCmd = [sumoBinary, '-c', self.path_to_sim_file]

        # A data structure which represents the roads conncetions
        self.road_structure = {}

        return

    def get_state_probabilistic_position(self):
        '''
        This function calculates a state of each intersection.
        The state representation is probabilistic position and described in the report.
        :return: A dictionary: {'intersection_name1': state,
                                'intersection_name2': state,...}
        '''
        env_state = {}

        for intersection in self.road_structure:
            lane_len = self.road_structure[intersection]['lane_len']
            num_of_phases = self.road_structure[intersection]['num_of_phases']
            num_of_lanes = self.road_structure[intersection]['num_of_lanes']
            x_offset = self.road_structure[intersection]['xy_position'][0]
            y_offset = self.road_structure[intersection]['xy_position'][1]

            # Prepare empty state of fixed size.
            # The state contains 8 discrete lines + one hot vec of phase.
            # TODO: consider moving this to parsing part. The empty state is same during all time...
            state = np.zeros(num_of_lanes * lane_len + num_of_phases, dtype=np.float32)

            for ilane, (lane_id, direction) in enumerate(self.road_structure[intersection]['lanes']):
                cars = traci.lane.getLastStepVehicleIDs(lane_id)
                lane_len = self.road_structure[intersection]['lane_len']

                # This loop iterates over all cars in current lane
                # and calculates a distance of the vehicle to the current
                # junction, i.e. a position relative to the junction.
                for icar in cars:
                    xcar, ycar = traci.vehicle.getPosition(icar)
                    if direction == 'vertical':
                        pos = abs(ycar - y_offset)
                    elif direction == 'horizontal':
                        pos = abs(xcar - x_offset)
                    else:
                        raise NotImplementedError

                    pos = np.clip(pos, 0., lane_len - 1. - 1e-6)
                    ipos = int(pos)

                    # Each car exist at the same time in two cells, with some certainty.
                    # In this way we can represent continuous position of a car on a discrete grid.
                    state[int(ilane * lane_len + ipos)] += 1. - (pos - ipos)
                    state[int(ilane * lane_len + ipos + 1)] += pos - ipos

                # Adding phase as one hot vector to state.
                num_of_phases = self.road_structure[intersection]['num_of_phases']
                state[num_of_lanes * lane_len: num_of_lanes * lane_len + num_of_phases] = \
                    np.eye(num_of_phases)[traci.trafficlight.getPhase(intersection)]

            env_state[intersection] = state
        return env_state

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

        state = self.get_state_probabilistic_position()

        reward, _ = self.calc_reward()

        if self.ncars > 250 or self.steps_done >= self.max_steps:
            done = True
            self.steps_done = 0

        return state, reward, done, np.array([[reward]])

    def calc_reward(self):
        '''
        The waiting time of a vehicle is defined as the time (in seconds) spent with a
        speed below 0.1m/s since the last time it was faster than 0.1m/s.
        (basically, the waiting time of a vehicle is reset to 0 every time it moves).
        A vehicle that is stopping intentionally with a <stop> does not accumulate waiting time. (c) Traci documentation.

        https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getWaitingTime

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
        try:
            traci.trafficlight.setProgram('gneJ00', '0')
        except:
            print("No program set.")

        self.parse_env()

        for _ in range(heatup):
            traci.simulationStep()
        return self.get_state_probabilistic_position()

    def parse_env(self):
        junctions = list(traci.trafficlight.getIDList())
        for junc in junctions:
            max_len = 0
            lanes_names = self._remove_repeating_lanes(list(traci.trafficlight.getControlledLanes(junc)))
            lanes_directions = []
            for lane in lanes_names:
                length = traci.lane.getLength(lane)
                if length > max_len:
                    max_len = length

                if traci.lane.getShape(lane)[0][0] == traci.lane.getShape(lane)[1][0]:
                    # X values the same -> vertical lane
                    lanes_directions.append('vertical')
                elif traci.lane.getShape(lane)[0][1] == traci.lane.getShape(lane)[1][1]:
                    # Y values the same -> horizontal
                    lanes_directions.append('horizontal')
                else:
                    lanes_directions.append('undefined')

            self.road_structure[junc] = {}
            self.road_structure[junc]['xy_position'] = traci.junction.getPosition(junc)
            self.road_structure[junc]['num_of_lanes'] = len(lanes_names)
            self.road_structure[junc]['lanes'] = tuple(zip(lanes_names, lanes_directions))
            self.road_structure[junc]['lane_len'] = ceil(max_len)

            num_of_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(junc)[0].phases)
            self.road_structure[junc]['num_of_phases'] = num_of_phases

            phase_list = []
            for idx in range(num_of_phases):
                phase_list.append((idx, traci.trafficlight.getCompleteRedYellowGreenDefinition(junc)[0].phases[idx].state))
            self.road_structure[junc]['phases_description'] = phase_list
        self.print_env()

    def _remove_repeating_lanes(self, lanes):
        '''
        Sometimes getControlledLanes(junc) returns a list with repeating elements.
        This is because one lane can have two lights controlling it. For example: Lights for going straight
        and a light for going right, from a single lane. This method removes repeating lanes.
        :param lanes: a list of strings.
        :return: a list of strings without repeating elements. The order is maintained.
        '''
        new_lanes = []
        for lane in lanes:
            if lane not in new_lanes:
                new_lanes.append(lane)
        return new_lanes

    def print_env(self):
        for junc in self.road_structure:
            print("========== Junction:", junc, " ==========")
            for key in self.road_structure[junc]:
                print('=== ', key, ' ===')
                print(self.road_structure[junc][key])

    def close(self):
        traci.close()
