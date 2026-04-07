import numpy as np
import random
import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete


class CircularRolloutEnv(gym.Env):
    """
    A decision tree environment.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
            self,
            reward_point = 2.,
            default_point = -1.,
            t_max = 30,
            cost = 0.01,
            aux_cost = 0.05,
            aux_cost_schedule = None,
            scale_factor = 1,
            seed = None,
        ):
        """
        Construct an environment.
        """

        self.reward_point = reward_point # reward point
        self.default_point = default_point # default point
        self.t_max = t_max # max time steps per episode
        self.cost = cost # cost per action
        self.scale_factor = scale_factor # reward scale facto

        # auxiliary penalty parameters
        self.aux_cost = aux_cost
        self.aux_cost_schedule = aux_cost_schedule
        self.global_step = 0

        # set random seed
        self.set_random_seed(seed)

        # initialize graph
        self.num_nodes = 9
        self.nodes = np.arange(9)

        # initialize stop features
        self.num_stop_features = 3
        self.stop_feature_dict = {
            0: 0, # yellow
            1: 1, # green
            2: 2, # red
            3: 0,
            4: 1,
            5: 2,
            6: 0,
            7: 1,
            8: 2,
        }

        # initialize reward features
        self.num_reward_features = 3
        self.reward_feature_dict = {
            0: 0, # vehicle
            1: 1, # animal
            2: 0,
            3: 2, # fruit
            4: 0,
            5: 1,
            6: 1,
            7: 2,
            8: 2,
        }

        # initialize action space
        self.action_space = Discrete(self.num_nodes + 2) # rollout, reset, accept, reject

        # initialize observation space
        observation_shape = (
            self.num_nodes + # fixation node (num_nodes,)
            self.num_nodes + # start node (num_nodes,)
            self.num_reward_features + # fixated reward feature (num_reward_features,)
            self.num_stop_features + # fixated stop feature (num_stop_features,)
            1 + # if reward based on current rule (1,)
            1, # if stop based on current rule (1,)
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = observation_shape)


    def reset(self):
        """
        Reset the environment.
        """

        # reset the trial
        self.init_trial()

        # get observation
        obs = self.get_obs()
    
        # get info
        info = {
            'mask': self.get_action_mask(),
        }

        return obs, info


    def step(self, action):
        """
        Step the environment.
        """

        action = int(action)

        self.time_elapsed += 1
        done = False
        reward = -self.cost # initialize reward as cost

        # simulate
        if action < 9:
            self.fixation_node = action

            # panalty
            if self.is_past_stop(action):
                reward -= self.aux_cost

                # update panalty
                if self.aux_cost_schedule is not None:
                    self.update_aux_cost()

        # accept
        elif action == 9:
            reward = self.compute_cumulative_reward() * self.scale_factor

        # reject
        elif action == 10:
            reward = 0.

        # done
        if action in [9, 10] or self.time_elapsed == self.t_max:
            # finish the trial
            done = True

        # get observation
        obs = self.get_obs()
    
        # get info
        info = {
            'mask': self.get_action_mask(),
        }

        return obs, reward, done, False, info
    
    
    def init_trial(self):
        """
        Initialize a trial.
        """

        # initialize time elapsed
        self.time_elapsed = 0

        # sample start node
        self.start_node = np.random.choice(self.nodes)

        # sample features of the trial
        self.reward_feature = np.random.choice(np.arange(self.num_reward_features))
        self.stop_feature = np.random.choice(np.arange(self.num_stop_features))

        # initialize fixation node
        self.fixation_node = self.start_node.copy()

        # pre-compute the true rollout path for this trial (used by aux penalty)
        self.rollout_path = self.compute_rollout_path()
    

    def compute_rollout_path(self):
        """
        Compute the ordered list of nodes visited in a correct forward rollout,
        starting from start_node and stopping (inclusive) at the first stop node.
 
        Returns:
            path: list of int. nodes on the true rollout path, including start ans stop node.
        """
 
        path = [self.start_node]
        node = self.start_node
 
        while self.get_stop(node) != 1:
            node = self.transition(node)
            path.append(node)
 
        return path

    
    def is_past_stop(self, node):
        """
        Check if node is on the path.
        """
 
        return node not in self.rollout_path
    

    def update_aux_cost(self):
        """
        Update auxiliary cost
        """

        self.global_step += 1

        if self.global_step < len(self.aux_cost_schedule):
            self.aux_cost = self.aux_cost_schedule[self.global_step]
        else:
            self.aux_cost = 0.


    def transition(self, node):
        """
        Transit to the next node.
        """

        node_next = (node + 1) % self.num_nodes

        return node_next


    def compute_cumulative_reward(self):
        """
        Compute cumulative reward.
        """

        # initialize cumulative reward and node
        cum_reward = 0.
        node = self.start_node

        while self.get_stop(node) != 1:
            # transit to the next node
            node = self.transition(node)

            # accumulate reward
            cum_reward += self.get_reward(node)
        
        return cum_reward


    def get_reward(self, node):
        """
        Get reward based on the node and reward feature.
        """

        # always return 0 for the start node
        if node == self.start_node:
            return self.default_point

        feature = self.reward_feature_dict[node]

        if feature == self.reward_feature:
            return self.reward_point
        else:
            return self.default_point
    

    def get_stop(self, node):
        """
        Get stop based on the node and stop feature.
        """

        # always return 0 for the start node
        if node == self.start_node:
            return 0

        feature = self.stop_feature_dict[node]

        if feature == self.stop_feature:
            return 1
        else:
            return 0
    

    def get_obs(self):
        """
        Get observation.
        """

        fixated_reward_feature = self.reward_feature_dict[self.fixation_node]
        fixated_stop_feature = self.stop_feature_dict[self.fixation_node]

        # wrap observation
        obs = np.hstack([
            self.one_hot_coding(num_classes = self.num_nodes, labels = self.fixation_node),
            self.one_hot_coding(num_classes = self.num_nodes, labels = self.start_node),
            self.one_hot_coding(num_classes = self.num_reward_features, labels = fixated_reward_feature),
            self.one_hot_coding(num_classes = self.num_stop_features, labels = fixated_stop_feature),
            self.get_reward(self.fixation_node),
            self.get_stop(self.fixation_node),
            # self.time_elapsed,
        ])

        return obs
    

    def get_action_mask(self):
        """
        Get action mask.

        Note:
            no batching is considered here. batching is implemented by vectorization wrapper.
            if no batch training is used, add the batch dimension and transfer the mask to torch.tensor in trainer.
            if batch training is used, concatenate batches and transfer the mask to torch.tensor in trainer.
        """

        mask = np.ones((self.action_space.n,), dtype = bool)

        return mask


    def set_random_seed(self, seed):
        """
        Set random seed.
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)


    def one_hot_coding(self, num_classes, labels = None):
        """
        One-hot code nodes.
        """

        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]

        return labels_one_hot



class MetaLearningWrapper(Wrapper):
    """
    A meta-RL wrapper.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, env):
        """
        Construct an wrapper.
        """

        super().__init__(env)

        self.env = env
        self.num_nodes = env.get_wrapper_attr('num_nodes')
        self.one_hot_coding = env.get_wrapper_attr('one_hot_coding')

        # initialize previous variables
        self.init_prev_variables()

        # define new observation space
        new_observation_shape = (
            self.env.observation_space.shape[0] + # obs
            self.env.action_space.n + # previous action
            1, # previous reward
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = new_observation_shape)


    def step(self, action):
        """
        Step the environment.
        """

        obs, reward, done, truncated, info = self.env.step(action)

        # concatenate previous variables into observation
        obs_wrapped = self.wrap_obs(obs)

        # update previous variables
        self.prev_action = action
        self.prev_reward = reward

        return obs_wrapped, reward, done, truncated, info
    

    def reset(self, seed = None, options = {}):
        """
        Reset the environment.
        """

        obs, info = self.env.reset()

        # initialize previous physical action and reward
        self.init_prev_variables()

        # concatenate previous physical action and reward into observation
        obs_wrapped = self.wrap_obs(obs)

        return obs_wrapped, info
    

    def init_prev_variables(self):
        """
        Reset previous variables.
        """

        self.prev_action = None
        self.prev_reward = 0.


    def wrap_obs(self, obs):
        """
        Wrap observation with previous variables.
        """

        obs_wrapped = np.hstack([
            obs, # current obs
            self.one_hot_coding(num_classes = self.env.action_space.n, labels = self.prev_action),
            self.prev_reward,
        ])
        return obs_wrapped
    



if __name__ == '__main__':
    # test single environment

    import warnings
    warnings.filterwarnings('ignore')
    
    env = CircularRolloutEnv()
    env = MetaLearningWrapper(env)

    for i in range(50):

        obs, info = env.reset()
        done = False

        print('initial obs:', obs.shape)
        
        while not done:
            # sample action
            # action = env.action_space.sample()
            valid_actions = np.where(info['mask'])[0]
            action = random.choice(valid_actions)

            # step env
            obs, reward, done, truncated, info = env.step(action)

            print(
                'action:', action, '|',
                'reward:', np.round(reward, 3), '|',
                'obs:', obs.shape, '|',
                'fixation node:', env.env.fixation_node, '|',
                'time elapsed:', env.env.time_elapsed, '|',
                'done:', done, '|',
            )
        print()