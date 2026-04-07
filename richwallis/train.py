import os
import gymnasium as gym

from modules import *


if __name__ == '__main__':

    # parse args
    parser = ArgParser()
    args = parser.args

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{args.beta_e_final}_{args.jobid}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # set environment
    seeds = [random.randint(0, 1000) for _ in range(args.batch_size)]
    env = gym.vector.SyncVectorEnv([
        lambda: MetaLearningWrapper(
            BanditEnv(
                num_bandits = args.num_bandits,
                value_min = args.value_min,
                value_max = args.value_max,
                value_mean = args.value_mean,
                value_std = args.value_std,
                reward_std = args.reward_std,
                noise_free_obs = args.noise_free_obs,
                t_wait = args.t_wait,
                t_max = args.t_max,
                stay_cost = args.stay_cost,
                switch_cost = args.switch_cost,
                scale_factor = args.scale_factor,
                seed = seeds[i],
            )
        )
        for i in range(args.batch_size)
    ])

    # set net
    net = SharedGRURecurrentActorCriticPolicy(
        feature_size = env.single_observation_space.shape[0],
        action_size = env.single_action_space.n,
        hidden_size = args.hidden_size,
    )

    # set model
    model = BatchMaskA2C(
        net = net,
        env = env,
        lr = args.lr,
        batch_size = args.batch_size,
        max_grad_norm = args.max_grad_norm,
        gamma = args.gamma,
        lamda = args.lamda,
        beta_v = args.beta_v,
        beta_e = args.beta_e,
        entropy_schedule = np.linspace(
            args.beta_e_init,
            args.beta_e_final,
            int(args.num_episodes / args.batch_size),
        )
    )

    # train network
    data = model.learn(
        num_episodes = args.num_episodes,
        print_frequency = 2000,
    )

    # save net and data
    model.save_net(os.path.join(exp_path, f'net.pth'))
    model.save_data(os.path.join(exp_path, f'data_training.p'))