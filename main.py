from experiments import Experiment
from rl_args import fixed_q_targets_single_intersection_args
from rl_args import fixed_q_targets_double_intersection_args
from rl_args import double_dqn_single_intersection_args
from rl_args import double_dqn_double_intersection_args

### disable some plt warnings.
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
###

def main():
    # # Experiment 1
    args = fixed_q_targets_single_intersection_args()
    args.target_update = 50
    exp = Experiment(args)
    exp.train_default()
    args = fixed_q_targets_single_intersection_args()
    args.target_update = 0.001
    exp = Experiment(args)
    exp.train_default()

    # # Experiment 2
    args = fixed_q_targets_double_intersection_args()
    args.target_update = 50
    exp = Experiment(args)
    exp.train_default()
    args = fixed_q_targets_double_intersection_args()
    args.target_update = 0.001
    exp = Experiment(args)
    exp.train_default()

    # # Experiment 3
    args = double_dqn_single_intersection_args()
    args.target_update = 50
    exp = Experiment(args)
    exp.train_default()
    args = double_dqn_single_intersection_args()
    args.target_update = 0.001
    exp = Experiment(args)
    exp.train_default()

    # Experiment 4
    args = double_dqn_double_intersection_args()
    args.target_update = 50
    exp = Experiment(args)
    exp.train_default()
    args = double_dqn_double_intersection_args()
    args.grad_clip = True
    exp = Experiment(args)
    exp.train_default()

main()