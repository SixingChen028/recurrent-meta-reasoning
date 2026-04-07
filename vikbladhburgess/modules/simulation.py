import numpy as np
import random
import pickle
import torch
import warnings
warnings.filterwarnings('ignore')

from .utils import *


def simulate(
        net,
        env,
        num_trials,
        greedy = False,
        include_hidden = False,
        include_logits = False,
        include_policy = False,
    ):
    """
    Simulate.
    """

    # reset environment
    env.reset()

    # reset data
    data = {
        'start_nodes': [],
        'reward_features': [],
        'stop_features': [],
        'fixation_seqs': [],
        'action_seqs': [],
    }
    if include_hidden:
        data['hidden_seqs'] = []
    if include_logits:
        data['logits_seqs'] = []
    if include_policy:
        data['policy_seqs'] = []

    # get net type
    net_type = type(net).__name__

    # iterate through trials
    for _ in range(num_trials):

        # initialize trial recordings
        fixation_seq_ep = []
        action_seq_ep = []
        if include_hidden:
            hidden_seq_ep = []
        if include_logits:
            logits_seq_ep = []
        if include_policy:
            policy_seq_ep = []

        # initialize a trial
        done = False
        states = None

        # reset environment
        obs, info = env.reset()
        obs = torch.Tensor(obs).unsqueeze(dim = 0) # (1, feature_dim)
        action_mask = torch.tensor(info['mask']) # (action_dim,)

        with torch.no_grad():
            # iterate through a trial
            while not done:

                # step the net
                action, policy, log_prob, entropy, value, states = net(
                    obs, states, action_mask
                )
                if greedy:
                    action = torch.argmax(policy)

                # step the env
                obs, reward, done, truncated, info = env.step(action.item())
                obs = torch.Tensor(obs).unsqueeze(dim = 0) # (1, feature_dim)
                action_mask = torch.tensor(info['mask']) # (action_dim,)

                # record results for the timestep
                fixation_seq_ep.append(env.env.fixation_node)
                action_seq_ep.append(int(action))
                if include_hidden:
                    hidden_seq_ep.append(process_hidden(states, net_type))
                if include_logits:
                    logits_seq_ep.append(net.policy_net.logits.squeeze().tolist())
                if include_policy:
                    policy_seq_ep.append(policy.squeeze().tolist())

            # record results for the trial
            data['start_nodes'].append(int(env.env.start_node))
            data['reward_features'].append(int(env.env.reward_feature))
            data['stop_features'].append(int(env.env.stop_feature))
            data['fixation_seqs'].append(fixation_seq_ep)
            data['action_seqs'].append(action_seq_ep)
            if include_hidden:
                data['hidden_seqs'].append(hidden_seq_ep)
            if include_logits:
                data['logits_seqs'].append(logits_seq_ep)
            if include_policy:
                data['policy_seqs'].append(policy_seq_ep)
    
    return data


def preprocess(data, args, merge_fixations = False):
    """
    Preprocess data.
    """

    num_trials = len(data['action_seqs'])

    # add variables
    data['lengths'] = []
    data['decisions'] = []

    for i in range(num_trials):
        
        action_seq_ep = pull(data, i, 'action_seqs')[0]

        data['lengths'].append(len(action_seq_ep))
        data['decisions'].append(action_seq_ep[-1])

    return data


def process_hidden(states, net_type):
    """
    Get hidden state.
    """

    if net_type == 'LSTMRecurrentActorCriticPolicy':
        hidden_processed = torch.cat((states[0][0], states[1][0]), axis = -1).squeeze().tolist() # (2 * num_hidden,)
    elif net_type == 'SharedLSTMRecurrentActorCriticPolicy':
        hidden_processed = states[0].squeeze().tolist() # (num_hidden,)
    elif net_type == 'GRURecurrentActorCriticPolicy':
        hidden_processed = torch.cat((states[0], states[1]), axis = -1).squeeze().tolist() # (2 * num_hidden,)
    elif net_type == 'SharedGRURecurrentActorCriticPolicy':
        hidden_processed = states.squeeze().tolist() # (num_hidden,)
    
    return hidden_processed


def pull(data, index, *keys):
    """
    Pull data according to keys.
    """
    return [data[key][index] for key in keys]


def save_data(data, path):
    """
    Save data.
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_data(path):
    """
    Load data.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data

