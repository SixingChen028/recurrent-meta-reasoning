import numpy as np
import random

import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete

from .graph import *
# from graph import * # debugging use


class DecisionTreeEnv(gym.Env):
    """
    A decision tree environment.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
            self,
            num_nodes = 11,
            t_max = 100,
            cost = 0.01,
            scale_factor = 1 / 8,
            shuffle_nodes = True,
            mask_fixation = False,
            seed = None,
        ):
        """
        Construct an environment.
        """

        self.num_nodes = num_nodes # number of nodes
        self.t_max = t_max # max time steps per episode
        self.cost = cost # cost per action
        self.scale_factor = scale_factor # reward scale factor
        self.shuffle_nodes = shuffle_nodes # if shuffle nodes
        self.mask_fixation = mask_fixation # if use fixation mask

        # set random seed
        self.set_random_seed(seed)

        # initialize point set
        self.point_set = np.array([-8, -4, -2, -1, 1, 2, 4, 8])

        # initialize graph
        self.graph = Graph(self.num_nodes, self.point_set)

        # initialize action space
        self.action_space = Discrete(self.num_nodes + self.num_nodes + 1)

        # initialize observation space
        observation_shape = (
            self.num_nodes + # fixation node (num_nodes,)
            self.num_nodes * 3 + # parent and childs of fixation node (3 * num_nodes,)
            self.num_nodes + # physical node (num_nodes,)
            3, # fixation point, time elapsed, stage
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = observation_shape,)


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
            'stage': self.stage,
            'mask': self.get_action_mask(),
        }

        return obs, info


    def step(self, action):
        """
        Step the environment.
        """

        self.time_elapsed += 1
        done = False
        reward = -self.cost # initialize reward as cost

        # get type action and node action
        action_type = int(action) // self.num_nodes # ensure int
        action_node = int(action) % self.num_nodes

        # fixation stage
        if self.stage == 0:
            # fixation action
            if action_type == 0:
                # move fixation to the node
                self.fixation_node = action_node

                # update record
                updated = self.update_fixation_record(action_node)

            # decision action (debugging use)
            elif action_type == 1:
                raise ValueError('Execute decision action in fixation stage.')

            # stage switch action
            elif action_type == 2:
                # switch to decision stage
                self.stage = 1

                # reset fixation to root node
                self.fixation_node = self.graph.root_node

        # decision stage
        elif self.stage == 1:
            # fixation action (debugging use)
            if action_type == 0:
                raise ValueError('Execute fixation action in decision stage.')

            # decision action
            elif action_type == 1:
                # move fixation to the node
                self.fixation_node = action_node

                # move physical node
                self.physical_node = action_node

                # add reward
                reward += self.get_scaled_reward()[self.physical_node]
            
            # stage switch action (debugging use)
            elif action_type == 2:
                raise ValueError('Execute stage switch action in decision stage.')

        # if reach a leaf node within time limit
        if self.stage == 1 and self.physical_node in self.graph.leaf_nodes:
            done = True

        # if time out
        elif self.time_elapsed == self.t_max:
            done = True

            # initialize terminal node as the current fixation node
            terminal_node = self.physical_node

            # choose a random trajectory
            while terminal_node not in self.graph.leaf_nodes:
                # randomly sample a terminal node
                terminal_node = random.sample(self.graph.successors(terminal_node), 1)[0]

                # add reward
                reward += self.get_scaled_reward()[terminal_node]

        # get observation
        obs = self.get_obs()
    
        # get info
        info = {
            'stage': self.stage,
            'mask': self.get_action_mask(),
        }

        return obs, reward, done, False, info
    
    
    def init_trial(self):
        """
        Initialize a trial.
        """

        # initialize time_elapsed and stage
        self.time_elapsed = 0
        self.stage = 0

        # initialize the tree
        self.graph.reset(shuffle_nodes = self.shuffle_nodes)

        # initialize fixation record
        self.init_fixation_record()

        # initialize fixation node and physical node to root node
        self.fixation_node = self.graph.root_node
        self.physical_node = self.graph.root_node
    

    def init_fixation_record(self):
        """
        Initialize fixation record.
        """

        # initialize node recordings
        self.visited_nodes = np.array([self.graph.root_node])
        self.candidate_nodes = np.array(self.graph.child_dict[self.graph.root_node])
        self.valid_nodes = np.union1d(self.visited_nodes, self.candidate_nodes)

    
    def update_fixation_record(self, node):
        """
        Update fixation record.
        """

        updated = False

        # if fixating a new candiate node
        if node in self.candidate_nodes:
            updated = True

            # include the new node into visited nodes
            self.visited_nodes = np.append(self.visited_nodes, node)

            # remove the new node from candidate nodes
            self.candidate_nodes = np.delete(self.candidate_nodes, np.where(self.candidate_nodes == node)[0])

            # if the new node has children, add them into candidate nodes
            if node in self.graph.child_dict.keys():
                self.candidate_nodes = np.append(self.candidate_nodes, self.graph.child_dict[node])
                    
        # update valid nodes
        self.valid_nodes = np.union1d(self.visited_nodes, self.candidate_nodes)

        return updated
    

    def get_obs(self):
        """
        Get observation.
        """

        # get parent and child nodes
        fixation_parent_node = self.graph.predecessors(self.fixation_node)
        fixation_child_nodes = self.graph.successors(self.fixation_node)

        # wrap observation
        obs = np.hstack([
            self.one_hot_coding(num_classes = self.num_nodes, labels = self.fixation_node),
            self.one_hot_coding(num_classes = self.num_nodes, labels = fixation_parent_node),
            self.one_hot_coding(num_classes = self.num_nodes, labels = fixation_child_nodes[0]),
            self.one_hot_coding(num_classes = self.num_nodes, labels = fixation_child_nodes[1]),
            self.one_hot_coding(num_classes = self.num_nodes, labels = self.physical_node),
            self.graph.points[self.fixation_node],
            self.time_elapsed,
            self.stage,
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

        mask = np.zeros((self.action_space.n,), dtype = bool)

        # fixation stage
        if self.stage == 0:
            if self.mask_fixation:
                # valid fixation actions are allowed
                mask[self.valid_nodes] = True
            else:
                # all fixation actions are allowed
                mask[:self.num_nodes] = True

            # stage fixation action is allowed
            mask[-1] = True

        # decision stage
        elif self.stage == 1:

            # get valid children
            valid_children = np.array(self.graph.successors(self.physical_node))

            # if have valid children
            if valid_children[0] != None:
                # valid decision actions are allowed
                mask[self.num_nodes + valid_children] = True
        
        return mask
    

    def get_scaled_reward(self):
        """
        Get scaled reward.
        """

        # compute rewards from points
        rewards = self.graph.points * self.scale_factor

        return rewards
    

    def get_scaled_cum_reward(self):
        """
        Get scaled cumulative reward.
        """

        # compute cumulative rewards from cumulative points
        cum_rewards = self.graph.cum_points * self.scale_factor

        return cum_rewards


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
    
    env = DecisionTreeEnv()
    env = MetaLearningWrapper(env)

    for i in range(50):

        obs, info = env.reset()
        done = False

        print('tree:', env.env.graph.child_dict)
        print('points:', env.env.graph.points)
        print('cumulative points:', env.env.graph.cum_points)
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
                'stage:', env.env.stage, '|',
                'fixation node:', env.env.fixation_node, '|',
                'physical node:', env.env.physical_node, '|',
                'time elapsed:', env.env.time_elapsed, '|',
                'done:', done, '|',
            )
        print()