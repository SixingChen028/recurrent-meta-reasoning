import numpy as np
import random
from scipy.stats import norm

import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete


class BanditEnv(gym.Env):
    """
    A bandit environment.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
            self,
            num_bandits = 2,
            value_min = -10,
            value_max = 10,
            value_mean = 0.0,
            value_std = 4.5,
            reward_std = 6.5,
            noise_free_obs = False,
            t_max = 100,
            stay_cost = 0.01,
            switch_cost = 0.025,
            scale_factor = 1 / 5,
            seed = None,
        ):
        """
        Construct an environment.
        """

        # set random seed
        self.set_random_seed(seed)

        # initialize parameters
        self.num_bandits = num_bandits # number of bandits
        self.value_set = np.arange(value_min, value_max + 1) # set of values
        self.value_mean = value_mean # value mean
        self.value_std = value_std # value std
        self.reward_std = reward_std # std of reward observation
        self.noise_free_obs = noise_free_obs # if include noise in observation
        self.t_max = t_max # max time steps per episode
        self.stay_cost = stay_cost # stay cost
        self.switch_cost = switch_cost # switch cost
        self.scale_factor = scale_factor # reward scale factor

        # initialize action space
        self.action_space = Discrete(self.num_bandits + self.num_bandits)

        # initialize observation space
        observation_shape = (
            self.num_bandits + # fixation bandit (num_bandits,)
            self.num_bandits + # physical bandit (num_bandits,)
            1, # reward obs
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = observation_shape,)

        # initialize sampling probs
        self.probs = norm.pdf(self.value_set, loc = self.value_mean, scale = self.value_std)
        self.probs /= self.probs.sum()


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
            'reward_obs': self.reward_obs,
            'item': self.fixation_bandit,
            'posterior_means': self.posterior_means.copy(),
            'posterior_precisions': self.posterior_precisions.copy(),
            'mask': self.get_action_mask(),
        }

        return obs, info


    def step(self, action):
        """
        Step the environment.
        """

        self.time_elapsed += 1
        done = False
        reward = 0. # initialize reward as 0

        # get type action and node action
        action_type = int(action) // self.num_bandits # ensure int
        action_bandit = int(action) % self.num_bandits

        # if fixation
        if action_type == 0:
            # first fixation gets stay cost
            if self.fixation_bandit == None:
                reward -= self.stay_cost * self.scale_factor
            
            # stay fixation gets stay cost
            elif self.fixation_bandit == action_bandit:
                reward -= self.stay_cost * self.scale_factor
            
            # switch fixation gets switch cost
            elif self.fixation_bandit != action_bandit:
                reward -= self.switch_cost * self.scale_factor

            # move fixation to the bandit
            self.fixation_bandit = action_bandit

            # sample reward observation
            if self.noise_free_obs:
                self.sample_noise_free_reward_obs()
            else:
                self.sample_reward_obs()

            # update belief
            self.update_belief(bandit = self.fixation_bandit, obs = self.reward_obs)
        
        # if decision
        elif action_type == 1:
            # move fixation to the bandit
            self.fixation_bandit = action_bandit

            # move physical bandit
            self.physical_bandit = action_bandit

            # add reward
            reward += self.values[self.physical_bandit] * self.scale_factor

        # if make a decision within time limit
        if action_type == 1:
            done = True

        # if time out
        elif self.time_elapsed == self.t_max:
            done = True

            # choose a random bandit
            random_bandit = random.choice(np.arange(0, self.num_bandits))

            # add reward
            reward += self.values[random_bandit] * self.scale_factor

        # get observation
        obs = self.get_obs()
    
        # get info
        info = {
            'reward_obs': self.reward_obs,
            'item': self.fixation_bandit,
            'posterior_means': self.posterior_means.copy(),
            'posterior_precisions': self.posterior_precisions.copy(),
            'mask': self.get_action_mask(),
        }

        return obs, reward, done, False, info
    
    
    def init_trial(self):
        """
        Initialize a trial.
        """

        # initialize time_elapsed
        self.time_elapsed = 0

        # get values
        self.values = np.random.choice(self.value_set, size = self.num_bandits, replace = True, p = self.probs)

        # initialize fixation bandit and physical bandit
        self.fixation_bandit = None
        self.physical_bandit = None

        # initialize reward obs 
        self.reward_obs = 0.

        # initialize belief
        self.posterior_means = np.full(self.num_bandits, self.value_mean)
        self.posterior_precisions = np.full(self.num_bandits, 1.0 / self.value_std ** 2)
    

    def sample_reward_obs(self):
        """
        Sample a reward observation.
        """

        self.reward_obs = np.random.normal(loc = self.values[self.fixation_bandit], scale = self.reward_std)
        
        return self.reward_obs
    

    def sample_noise_free_reward_obs(self):
        """
        Sample a reward observation without noise.
        """

        self.reward_obs = self.values[self.fixation_bandit]
        
        return self.reward_obs
    

    def update_belief(self, bandit, obs):
        """
        Update posterior belief of a specific bandit using Gaussian conjugate update.
        """

        prior_mu = self.posterior_means[bandit]
        prior_tau = self.posterior_precisions[bandit]
        likelihood_tau = 1.0 / self.reward_std ** 2

        # Bayesian update
        posterior_tau = prior_tau + likelihood_tau
        posterior_mu = (prior_mu * prior_tau + obs * likelihood_tau) / posterior_tau

        self.posterior_means[bandit] = posterior_mu
        self.posterior_precisions[bandit] = posterior_tau
    

    def get_obs(self):
        """
        Get observation.
        """

        # wrap observation
        obs = np.hstack([
            self.one_hot_coding(num_classes = self.num_bandits, labels = self.fixation_bandit),
            self.one_hot_coding(num_classes = self.num_bandits, labels = self.physical_bandit),
            self.reward_obs,
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
        self.num_bandits = env.get_wrapper_attr('num_bandits')
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
    
    env = BanditEnv(
        num_bandits = 2,
        value_min = -10.,
        value_max = 10.,
        reward_std = 3.,
        t_max = 50,
        stay_cost = 0.01,
        switch_cost = 0.025,
        scale_factor = 1 / 5,
    )
    env = MetaLearningWrapper(env)

    for i in range(50):

        obs, info = env.reset()
        done = False

        print('reward means:', env.env.values)
        print('initial obs:', obs.shape)
        
        while not done:
            # sample action
            action = env.action_space.sample()

            # step env
            obs, reward, done, truncated, info = env.step(action)

            print(
                'action:', action, '|',
                'reward:', np.round(reward, 3), '|',
                'obs:', obs.shape, '|',
                'fixation bandit:', env.env.fixation_bandit, '|',
                'physical bandit:', env.env.physical_bandit, '|',
                'time_elapsed:', env.env.time_elapsed, '|',
                'done:', done, '|',
                info
            )
        print()