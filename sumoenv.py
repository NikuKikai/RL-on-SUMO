import os
import sys
import traci
import numpy as np
import random
from math import ceil
#import seaborn as sb
import matplotlib.pyplot as plt
from datetime import datetime

class SumoEnv:
    def __init__(self, args, path_to_sim_file='simulations\\intersection.sumocfg', gui=False):
        self.wt_last = 0.
        self.ncars = 0
        self.max_steps = args.sim_max_steps
        self.path_to_sim_file = path_to_sim_file

        # reward and state types
        self.state_type = args.state_type
        self.reward_type = args.reward_type

        # How much simulation steps perfromed in current episode.
        self.steps_done = 0
        # How much simulation steps perfromed each step_d call.
        self.sim_steps = args.sim_steps


        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        exe = 'sumo-gui.exe' if gui else 'sumo.exe'
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', exe)
        self.sumoCmd = [sumoBinary, '-c', self.path_to_sim_file, '--no-warnings'] #,'--no-step-log',

        # A data structure which represents the roads conncetions
        self.road_structure = {}
        self.parse_env()

        return

    def get_road_structure(self):
        return self.road_structure

    def get_dimensions(self):
        '''
        This function returns two important properties for each intersection:
        1) Size of the vector representing the state.
        2) number of available actions.
        :return: dictionary of tuples.
        '''
        dim_dict = {}
        for intersection in self.road_structure:
            num_of_actions = self.road_structure[intersection]['num_of_phases']
            num_of_lanes = self.road_structure[intersection]['num_of_lanes']
            lane_len = self.road_structure[intersection]['lane_len']

            if self.state_type == 'position':
                dim_dict[intersection] = (lane_len * num_of_lanes + num_of_actions, num_of_actions)
            elif self.state_type == 'density':
                dim_dict[intersection] = (num_of_lanes + num_of_actions, num_of_actions)
            elif self.state_type == 'density_and_speed':
                dim_dict[intersection] = (2 * num_of_lanes + num_of_actions, num_of_actions)
            else:
                raise NotImplementedError

        return dim_dict

    def get_state(self):
        if self.state_type == 'density':
            return self._get_state_density()
        elif self.state_type == 'position':
            return self._get_state_probabilistic_position()
        elif self.state_type == 'density_and_speed':
            return self._get_state_density_and_speed()
        else:
            raise NotImplementedError

    def _get_state_density_and_speed(self):
        '''
        This function calculates a state of each intersection.
        The state representation is lane occupancy.
        https://sumo.dlr.de/pydoc/traci._lane.html#LaneDomain-getLastStepOccupancy
        and a avg lane speed.
        https://sumo.dlr.de/pydoc/traci._lane.html#LaneDomain-getLastStepMeanSpeed
        :return: A dictionary: {'intersection_name1': state,
                                'intersection_name2': state,...}
        '''
        env_state = {}

        for intersection in self.road_structure:
            num_of_phases = self.road_structure[intersection]['num_of_phases']
            num_of_lanes = self.road_structure[intersection]['num_of_lanes']

            # Prepare empty state of fixed size.
            # TODO: consider moving this to parsing part. The empty state is same during all time...
            state = np.zeros(2 * num_of_lanes + num_of_phases, dtype=np.float32)

            for ilane, (lane_id, _) in enumerate(self.road_structure[intersection]['lanes']):
                state[2 * ilane] = traci.lane.getLastStepOccupancy(lane_id) # add density
                state[2 * ilane + 1] = traci.lane.getLastStepMeanSpeed(lane_id) # add mean speed

            # Adding phase as one hot vector to state.
            state[2 * num_of_lanes: 2 * num_of_lanes + num_of_phases] = \
                    np.eye(num_of_phases)[traci.trafficlight.getPhase(intersection)]

            env_state[intersection] = state
        return env_state

    def _get_state_density(self):
        '''
        This function calculates a state of each intersection.
        The state representation is lane occupancy.
        https://sumo.dlr.de/pydoc/traci._lane.html#LaneDomain-getLastStepOccupancy
        :return: A dictionary: {'intersection_name1': state,
                                'intersection_name2': state,...}
        '''
        env_state = {}

        for intersection in self.road_structure:
            num_of_phases = self.road_structure[intersection]['num_of_phases']
            num_of_lanes = self.road_structure[intersection]['num_of_lanes']

            # Prepare empty state of fixed size.
            # TODO: consider moving this to parsing part. The empty state is same during all time...
            state = np.zeros(num_of_lanes + num_of_phases, dtype=np.float32)

            for ilane, (lane_id, _) in enumerate(self.road_structure[intersection]['lanes']):

                state[ilane] = traci.lane.getLastStepOccupancy(lane_id)

                # Adding phase as one hot vector to state.
                state[num_of_lanes: num_of_lanes + num_of_phases] = \
                    np.eye(num_of_phases)[traci.trafficlight.getPhase(intersection)]

            env_state[intersection] = state
        return env_state

    def _get_state_probabilistic_position(self):
        '''
        TODO: consider using https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getLanePosition instead
        TODO: The position of the vehicle along the lane
        TODO: (the distance from the front bumper to the start of the lane in [m]); error value: -2^30.

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
            state[num_of_lanes * lane_len: num_of_lanes * lane_len + num_of_phases] = \
                np.eye(num_of_phases)[traci.trafficlight.getPhase(intersection)]

            # Visialize the state:
            #if intersection == 'gneJ00':
            if False:
                state_pic = state[0: -4].reshape([num_of_lanes, -1])
                sb.heatmap(state_pic, center=0)
                plt.ion()
                plt.show()
                plt.pause(0.001)
                name = 'state_pictures\\pic' \
                       + str(datetime.now()).replace(':', '_').replace('-', '_').replace('.', '_').replace(' ', '_')\
                       + '.PNG'
                plt.savefig(name)
                plt.clf()

            env_state[intersection] = state
        return env_state

    def do_step(self, action_dict):
        '''
        This method applies an action on the environment, i.e. set a phase
        on each of the Intersections.
        :param action: A dictionary:    {'intersection_name1': 1,
                                        'intersection_name2': 3,...}
        :return: state - a dictionary:  {'intersection_name1': state,
                                        'intersection_name2': state,...},
                reward - a dictionary:  {'intersection_name1': reward,
                                        'intersection_name2': reward,...},
                done - either the simulation ended.
        '''
        self.steps_done += self.sim_steps
        done = False

        # Apply actions on intersection one by one.
        for intersection in self.road_structure:
            action = np.squeeze(action_dict[intersection])
            traci.trafficlight.setPhase(intersection, action)

        # Perform multiple steps of simulation.
        for _ in range(self.sim_steps):
            traci.simulationStep()

        # Get the new_state of all intersections.
        new_state = self.get_state()

        reward = self.calc_reward()

        if self.steps_done >= self.max_steps:
            done = True
            self.steps_done = 0

        return new_state, reward, done

    def calc_reward(self):
        '''
        This function calculates the reward of the action.
        The reward type is defined in the rl_args file.
        '''
        if self.reward_type == 'wt_sum_absolute':
            return self._calc_reward_wt_sum_absolute()
        elif self.reward_type == 'wt_avg_absolute':
            return self._calc_reward_wt_avg_absolute()
        elif self.reward_type == 'wt_sum_relative':
            return self._calc_reward_wt_sum_relative()
        elif self.reward_type == 'wt_max':
            return self._calc_reward_wt_max()
        elif self.reward_type == 'accumulated_wt_max':
            return self._calc_reward_accumulated_wt_max()
        elif self.reward_type == 'wt_squares_sum':
            return self._calc_reward_wt_squares_sum()
        elif self.reward_type == 'wt_parametric':
            return self._calc_reward_parametric()
        else:
            print("No reward type defined!!!")
            raise NotImplementedError

    def _calc_reward_parametric(self):
        '''
        This function calculates reward, based on parametrization of other reward functions.
        Currently the parameters are hardcoded but can be changed in future.
        :return: absolute_reward_dict - a dictionary:  {'intersection_name1': absolute_reward,
                                                        'intersection_name2': absolute_reward,...},
        '''
        absolute_reward_dict = {}
        for intersection in self.road_structure:
            wt = 0
            max_wt = 0
            for lane_id, _ in self.road_structure[intersection]['lanes']:
                # total waiting time for lane
                wt += traci.lane.getWaitingTime(lane_id)

                # find max waiting time for lane
                car_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                for car_id in car_ids:
                    curr_wt = traci.vehicle.getWaitingTime(car_id)
                    if max_wt < curr_wt:
                        max_wt = curr_wt

            # Calculate parametric reward
            absolute_reward = (-1)*((0.6 * wt) + (0.4 * max_wt))
            absolute_reward_dict[intersection] = absolute_reward
        return absolute_reward_dict

        wt_max_reward_dict = {}
        for intersection in self.road_structure:
            max_wt = 0
            for lane_id, _ in self.road_structure[intersection]['lanes']:
                car_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                for car_id in car_ids:
                    curr_wt = traci.vehicle.getWaitingTime(car_id)
                    if max_wt < curr_wt:
                        max_wt = curr_wt
            wt_max_reward = - max_wt
            wt_max_reward_dict[intersection] = wt_max_reward
        return wt_max_reward_dict


    def _calc_reward_wt_sum_absolute(self):
        '''
        This function calculates reward.
        Absolute sum of waiting times.
        :return: absolute_reward_dict - a dictionary:  {'intersection_name1': absolute_reward,
                                                        'intersection_name2': absolute_reward,...},
        '''
        absolute_reward_dict = {}
        for intersection in self.road_structure:
            wt = 0
            for lane_id, _ in self.road_structure[intersection]['lanes']:
                wt += traci.lane.getWaitingTime(lane_id)

            absolute_reward = - wt #/ 10e3  # scale factor
            absolute_reward_dict[intersection] = absolute_reward
        return absolute_reward_dict

    def _calc_reward_wt_avg_absolute(self):
        '''
        This function calculates reward.
        Absolute average waiting time per car.
        :return: absolute_reward_dict - a dictionary:  {'intersection_name1': absolute_reward,
                                                        'intersection_name2': absolute_reward,...},
        '''
        absolute_reward_dict = {}
        for intersection in self.road_structure:
            wt = 0
            for lane_id, _ in self.road_structure[intersection]['lanes']:
                cars_num = traci.lane.getLastStepVehicleNumber(lane_id)
                if cars_num != 0:
                    wt += traci.lane.getWaitingTime(lane_id)/cars_num

            absolute_reward = - wt #/ 10e3  # scale factor
            absolute_reward_dict[intersection] = absolute_reward
        return absolute_reward_dict


    def _calc_reward_wt_sum_relative(self):
        '''
        This function calculates reward. Relative wait time - An increase or decrease of wait
        time relatively to previous state.
        :return: relative_reward_dict - a dictionary:  {'intersection_name1': relative_reward,
                                                        'intersection_name2': relative_reward,...},
        '''
        relative_reward_dict = {}
        for intersection in self.road_structure:
            wt = 0
            for lane_id, _ in self.road_structure[intersection]['lanes']:
                wt += traci.lane.getWaitingTime(lane_id)
            relative_reward = - (wt - self.wt_last) * 50  # scale factor
            self.wt_last = wt
            relative_reward_dict[intersection] = relative_reward
        return relative_reward_dict

    def _calc_reward_accumulated_wt_max(self):
        '''
        This function calculates reward. The maximum accumulated waiting time across all vehicles for each intersection.
        https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime
        The problem with this reward, that in real world system, we can't gather this kind of data.
        :return: relative_reward_dict - a dictionary:  {'intersection_name1': wt_max_reward,
                                                        'intersection_name2': wt_max_reward,...},
        '''
        wt_max_reward_dict = {}
        for intersection in self.road_structure:
            max_wt = 0
            for lane_id, _ in self.road_structure[intersection]['lanes']:
                car_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                for car_id in car_ids:
                    curr_wt = traci.vehicle.getAccumulatedWaitingTime(car_id)
                    if max_wt < curr_wt:
                        max_wt = curr_wt
            wt_max_reward = - max_wt
            wt_max_reward_dict[intersection] = wt_max_reward
        return wt_max_reward_dict

    def _calc_reward_wt_max(self):
        '''
        This function calculates reward. The maximum waiting time across all vehicles for each intersection.
        https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getWaitingTime
        :return: relative_reward_dict - a dictionary:  {'intersection_name1': wt_max_reward,
                                                        'intersection_name2': wt_max_reward,...},
        '''
        wt_max_reward_dict = {}
        for intersection in self.road_structure:
            max_wt = 0
            for lane_id, _ in self.road_structure[intersection]['lanes']:
                car_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                for car_id in car_ids:
                    curr_wt = traci.vehicle.getWaitingTime(car_id)
                    if max_wt < curr_wt:
                        max_wt = curr_wt
            wt_max_reward = - max_wt
            wt_max_reward_dict[intersection] = wt_max_reward
        return wt_max_reward_dict

    def _calc_reward_wt_squares_sum(self):
        '''
        This function calculates reward. Square sum of waiting times across all vehicles for each intersection.
        https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getWaitingTime
        :return: relative_reward_dict - a dictionary:  {'intersection_name1': wt_max_reward,
                                                        'intersection_name2': wt_max_reward,...},
        '''
        wt_reward_dict = {}
        for intersection in self.road_structure:
            wt = 0
            for lane_id, _ in self.road_structure[intersection]['lanes']:
                car_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                for car_id in car_ids:
                    wt += traci.vehicle.getWaitingTime(car_id) ** 2
            wt_reward = - wt/10e8
            wt_reward_dict[intersection] = wt_reward
        return wt_reward_dict

    def parse_env(self):
        traci.start(self.sumoCmd, label='AI-project', )

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
                phase_list.append(
                    (idx, traci.trafficlight.getCompleteRedYellowGreenDefinition(junc)[0].phases[idx].state))
            self.road_structure[junc]['phases_description'] = phase_list
        self.print_env()

        traci.close()

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

    def reset(self, heatup=50):
        self.wt_last = 0.
        self.ncars = 0
        traci.start(self.sumoCmd, label='AI-project', )

        steps = random.randint(5, heatup)
        for _ in range(steps):
            traci.simulationStep()
        return self.get_state()

    def close(self):
        traci.close()
