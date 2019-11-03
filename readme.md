# Reinforcement Learning Traffic Light System 
Efficient traffic light control system may reduce traffic congestion, reduce CO2 emission and save expensive time. Most basic traffic light systems deploys fixed scheduling programs which are not considering real time traffic. Other traffic light systems may deploy fixed programs with real time priority for early defined vehicle types like public transportation. We propose efficient light control system based on reinforcement learning to lower traffic congestion by controlling traffic signals timing.
Traffic signal control can be modeled as a sequential decision-making problem, reinforcement learning can solve sequential decision-making problems by learning an optimal policy. Deep reinforcement
learning makes use of deep neural networks, powerful function approximators which benefit from large amounts of data.
The proposed system reduces delay, queues, vehicle stopped time
and travel time compared to conventional traffic signal controllers , concluding from that using reinforcement learning methods which explicitly develop the policy offers improved
performance over purely value-based methods.
### Project's goal
The goal of this project is optimizing the efficiency of intersection usage by changing timing and duration of each traffic light phase, based on current traffic state and previously learned data. The simplified goal can be defined as minimizing average waiting time in the intersection per vehicle, while the actual goal is to minimize weighted average delay, while each delay weighted by vehicle type such that some transportation types prioritized higher than others. This way emergency situations (Ambulance, Firefighters and Police) can be gracefully handled by the traffic light system without human intervention.

### Usage instructions
Arguments:
* --sumo-cfg: path to desired simulation configuration file, default: ./networks/double/Israel_zipping/network.sumocfg.
* --agent: RL agent to use, supported agents: [fixed_q_targets, double_dqn] , default: fixed_q_targets.
* --episodes: Number of episodes for simulation, default: agent\'s default value.
* --batch-size: Batch size for nn, default: agent\'s default value.
* --epsilon-decay: Epsilon decay rate, default: agent\'s default value.
* --state-type: State representation type, supported types: [density, position,
                             density_and_speed, density_speed_emergency, density_speed_bus,
                             density_speed_bus_emergency], default: agent\'s default type.
* --reward-type: Reward calculation type, supported types: [wt_sum_absolute, wt_avg_absolute,
                             wt_sum_relative, wt_max, accumulated_wt_max, wt_squares_sum,
                             wt_parametric, wt_vehicle_class], default: agent\'s default type.
* --nn-layers: NN layers, input example: 10 20 40 would be translated to [10, 20, 40],
                              default: agent\'s default type.
* --log-path: Root dir for result logs, default: agent\'s default type.
* --grad-clip: Grad clip value, default: agent\'s default value.
* --target-update: Each how many steps to update target nn, default: agent\'s default value.
* --capture-each: Each how many episodes to capture, default: agent\'s default value.
* --replay_size: Capacity of replay buffer, default: agent\'s default value.
* --sim-length: Simulation length in steps, default: agent\'s default value.
* --gui-mode: GUI mode: [diasble, enable, capture], default: disable

Usage: python ./main.py
Please use --help for additional information.

### Video captured example
The following clip executed by using reward type wt_total_acc_relative, state type density_queue_phases, and double_dqn agent with default additional parameters.
[Link to YouTube](https://youtu.be/G_7-YofVi0o)
 