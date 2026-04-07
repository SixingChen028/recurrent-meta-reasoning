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
    parser.write_args({
        'value_min': 0., # only simulate positive rewards
    })
    args = parser.args

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.num_bandits}_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{args.beta_e_final}_{args.jobid}')

    # load net
    net = torch.load(os.path.join(exp_path, f'net.pth'), weights_only = False)

    # set environment
    env = MetaLearningWrapper(
        BanditEnv(
            num_bandits = args.num_bandits,
            value_min = args.value_min,
            value_max = args.value_max,
            value_mean = args.value_mean,
            value_std = args.value_std,
            reward_std = args.reward_std,
            noise_free_obs = args.noise_free_obs,
            t_max = args.t_max,
            stay_cost = args.stay_cost,
            switch_cost = args.switch_cost,
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
        include_hidden = False,
        include_logits = False,
        include_policy = False,
    )
    save_data(data, os.path.join(exp_path, f'data_simulation.p'))


