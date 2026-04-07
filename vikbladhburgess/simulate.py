import numpy as np
import random
import torch
import warnings
warnings.filterwarnings('ignore')

from modules import *


if __name__ == '__main__':

    # set random seed
    seed = 15
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # parse args
    parser = ArgParser()
    args = parser.args

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.jobid}')

    # load net
    net = torch.load(os.path.join(exp_path, f'net.pth'), weights_only = False)

    # set environment
    env = MetaLearningWrapper(
        CircularRolloutEnv(
            reward_point = args.reward_point,
            default_point = args.default_point,
            t_max = args.t_max,
            cost = args.cost,
            aux_cost = 0.,
            scale_factor = args.scale_factor,
        )
    )

    # simulate
    num_trials = 100000
    data = simulate(
        net = net,
        env = env,
        num_trials = num_trials,
        greedy = False,
        include_hidden = True,
        include_logits = False,
        include_policy = False,
    )
    save_data(data, os.path.join(exp_path, f'data_simulation.p'))



