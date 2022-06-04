import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
import os
import numpy as np

from dodgy_env import DodgyEnv

def create_policy_network(observation_space, action_space, device='cpu'):
    return nn.Sequential(
        nn.Flatten(start_dim=-2, end_dim=-1),
        nn.Linear(
            torch.prod(torch.tensor(observation_space.shape)),
            32),
        nn.Tanh(),
        nn.Linear(32, action_space.n)).to(device)

def get_action_dist(policy_network, observation):
    logits = policy_network(observation)
    return Categorical(logits=logits)

def get_action(policy_network, observation):
    action_dist = get_action_dist(policy_network, observation)
    sample = action_dist.sample()
    return sample.item()

def compute_loss(policy_network, observation, action, weights):
    action_dist = get_action_dist(policy_network, observation)
    log_prob = action_dist.log_prob(action)
    return -(log_prob * weights).mean()

def run_one_epoch(env, policy_network, optimizer, *, render=False, min_batch_size=5000, inference_mode=False, device='cpu'):
    batch_observations = []
    batch_actions = []
    batch_weights = []
    batch_returns = []
    batch_lengths = []

    observation = env.reset()
    episode_done = False
    episode_rewards = []

    rendered_epoch = False

    epoch_done = False

    while not epoch_done:
        if (not rendered_epoch) and render:
            env.render()

        batch_observations.append(observation.copy())

        action = get_action(
            policy_network,
            torch.as_tensor(observation, dtype=torch.float32, device=device))
        observation, reward, episode_done, _ = env.step(action)

        batch_actions.append(action)

        episode_rewards.append(reward)

        if episode_done:
            episode_return, episode_length = sum(episode_rewards), len(episode_rewards)
            batch_returns.append(episode_return)
            batch_lengths.append(episode_length)

            batch_weights += [episode_return] * episode_length

            if (not rendered_epoch) and render:
                env.render()

            observation = env.reset()
            episode_done = False
            episode_rewards = []

            rendered_epoch = True

            if len(batch_observations) > min_batch_size:
                break
    if not inference_mode:
        optimizer.zero_grad()

        batch_observations_ = torch.as_tensor(
            np.array(batch_observations),
            dtype=torch.float32,
            device=device)

        batch_actions_ = torch.as_tensor(
            np.array(batch_actions),
            dtype=torch.int32,
            device=device)

        batch_weights_ = torch.as_tensor(
            np.array(batch_weights),
            dtype=torch.float32,
            device=device)

        batch_loss = compute_loss(
            policy_network,
            batch_observations_,
            batch_actions_,
            batch_weights_)

        batch_loss.backward()
        optimizer.step()

    else:
        batch_loss = None

    return batch_loss, batch_returns, batch_lengths
            


def run(env_type, *, lr=0.01, max_epochs=None, min_batch_size=5000, render=False, file_path=None, inference_mode=False):
    if not issubclass(env_type, gym.Env):
        raise TypeError(f'env_type must be a gym.Env')

    env = env_type()

    if not isinstance(env.observation_space, gym.spaces.Box):
        raise RuntimeError('Expected a continuous state space')

    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise RuntimeError('Expected a discrete action space')

    if inference_mode:
        if file_path is None:
            raise RuntimeError((
                'Cannot run inference mode without an existing trained policy. '
                'Please specify `file_path`'))
        elif not os.path.exists(file_path):
            raise RuntimeError((
                f'File "{file_path}" does not exist, and inference mode is '
                'turned on. You must either use an existing file or turn off '
                'inference mode to create a new one.'))

        policy_network = torch.load(file_path)

    else:
        # Load the policy from file if the file exists. Otherwise, create a
        # new policy network

        # If file path was given and the file exists, load the network from
        # the file. Otherwise, create a new one
        if file_path is not None and os.path.exists(file_path):
            policy_network = torch.load(file_path)

        else:
            policy_network = create_policy_network(env.observation_space, env.action_space)

    optimizer = Adam(policy_network.parameters(), lr=lr)

    i = 0
    while max_epochs is None or i < max_epochs:
        batch_loss, batch_returns, batch_lengths = run_one_epoch(
            env,
            policy_network,
            optimizer,
            render=render,
            min_batch_size=min_batch_size,
            inference_mode=inference_mode)

        if inference_mode:
            info = 'epoch: %3d loss: None return: %.3f ep_len: %.3f' % (i, np.mean(batch_returns), np.mean(batch_lengths))
        else:
            info = 'epoch: %3d loss: %.3f return: %.3f ep_len: %.3f' % (i, batch_loss, np.mean(batch_returns), np.mean(batch_lengths))
        print(info)


        # If not using inference mode and file path was given, save the
        # policy to the file
        if not inference_mode and file_path is not None:
            torch.save(policy_network, file_path)

        i += 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--render',
        action='store_true',
        help='render the environment at each step')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='learning rate for the optimizer')

    parser.add_argument(
        '--file_path',
        type=str,
        help='file to load/save trained policy network')

    parser.add_argument(
        '--inference_mode',
        action='store_true',
        help='run a trained policy without training')

    args = parser.parse_args()

    min_batch_size = 0 if args.inference_mode else 5_000

    run(DodgyEnv,
        lr=args.lr,
        max_epochs=None,
        min_batch_size=min_batch_size,
        render=args.render,
        file_path=args.file_path,
        inference_mode=args.inference_mode)

