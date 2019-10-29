from Environment.sumoenv import SumoEnv
from Environment.traffic_system_manager import TrafficSystemManager
import matplotlib.pyplot as plt
import os
from datetime import datetime

class Experiment():
    def __init__(self, args, show_training_curve=True):
        self.args = args
        self.show_training_curve = show_training_curve

        # Create folder for experiments results
        root_path = args.experiment_res_path
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        timestamp = str(datetime.now()).replace(':', '_').replace('-', '_').replace('.', '_').replace(' ', '_')
        sim_name = os.path.split(args.sim_file)[1].split('.')[0]
        folder_name = timestamp + '_' + args.rl_algo + '_' + sim_name
        self.res_dir_path = os.path.join(root_path, folder_name)
        os.makedirs(self.res_dir_path)

        # Create env.
        self.env = SumoEnv(self.args, capture_path=self.res_dir_path)

        # Save args
        self.args_file_path = os.path.join(self.res_dir_path, 'args.txt')
        with open(self.args_file_path, 'a') as file:
            for key in vars(args):
                file.write(str(key)+': '+str(vars(args)[key])+'\n')

    def train_default(self):
        manager = TrafficSystemManager(self.env.get_dimensions(), self.args)
        for epi in range(self.args.episodes):
            print("### Starting Episode: ", epi, ' ###')
            state = self.env.reset(heatup=self.args.sim_heatup)
            done = False
            while not done:
                action = manager.select_action(state)
                next_state, reward, done = self.env.do_step(action)
                manager.add_to_memory(state, action, next_state, reward)  # TODO: missing done
                manager.teach_agents()  # try to optimize if enough samples in memory.
                state = next_state

            manager.dump_data_on_episode_end(self.res_dir_path)
            self.env.close()
