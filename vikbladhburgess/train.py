import os
import gymnasium as gym

from modules import *


if __name__ == '__main__':

    # parse args
    parser = ArgParser()
    args = parser.args

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.jobid}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # set environment
    seeds = [random.randint(0, 1000) for _ in range(args.batch_size)]
    env = gym.vector.SyncVectorEnv([
        lambda: MetaLearningWrapper(
            CircularRolloutEnv(
                reward_point = args.reward_point,
                default_point = args.default_point,
                t_max = args.t_max,
                cost = args.cost,
                aux_cost = args.aux_cost,
                aux_cost_schedule = np.linspace(
                    args.cost,
                    0.,
                    args.num_episodes,
                ),
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
        kappa_squared = args.kappa_squared,
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
        # checkpoint_frequency = 50000,
        # checkpoint_path = exp_path,
    )

    # save net and data
    model.save_net(os.path.join(exp_path, f'net.pth'))
    model.save_data(os.path.join(exp_path, f'data_training.p'))