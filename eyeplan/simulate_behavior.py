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
    exp_path = os.path.join(args.path, f'exp_{args.cost}_{args.beta_e_final}_{args.kappa_squared}_{args.jobid}')

    # load net
    net = torch.load(os.path.join(exp_path, f'net.pth'), weights_only = False)

    # set environment
    env = MetaLearningWrapper(
        DecisionTreeEnv(
            num_nodes = args.num_nodes,
            t_max = args.t_max,
            cost = args.cost,
            scale_factor = args.scale_factor,
            shuffle_nodes = args.shuffle_nodes,
            mask_fixation = args.mask_fixation,
        )
    )

    # simulate
    num_trials = 100000
    data = simulate(
        net = net,
        env = env,
        num_trials = num_trials,
        greedy = False,
        include_hidden = False,
        include_logits = True,
        include_policy = False,
    )
    save_data(data, os.path.join(exp_path, f'data_simulation.p'))



