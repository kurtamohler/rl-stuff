import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from trench_runner import TrenchRunnerEnv
import os

device = 'cpu'

def my_mlp(observation_space, action_space):
    # Works well
    return nn.Sequential(
        nn.Flatten(start_dim=-2, end_dim=-1),
        nn.Linear(
            torch.prod(torch.tensor(observation_space.shape)),
            32),
        nn.Tanh(),
        nn.Linear(32, action_space.n)).to(device)


    # Slow and not sure if it works
    #return nn.Sequential(
    #    nn.Unflatten(-2, (1, 1, observation_space.shape[0])),
    #    nn.Flatten(start_dim=0, end_dim=-4),
    #    nn.Conv2d(1, 3, 5),
    #    #nn.Tanh(),
    #    nn.ReLU(),
    #    nn.Flatten(start_dim=-3, end_dim=-1),
    #    nn.Linear(5 * 5 * 3, action_space.n))

    # Not too good
    #return nn.Sequential(
    #    nn.Flatten(start_dim=-2, end_dim=-1),
    #    nn.Linear(
    #        torch.prod(torch.tensor(observation_space.shape)),
    #        16),
    #    nn.Tanh(),
    #    nn.Linear(16, 8),
    #    nn.Tanh(),
    #    nn.Linear(8, action_space.n)).to(device)

    # Adapted from "Deep Reinforcement Learning in Action"
    # Seems like it's crap for this application.
    # But the original uses learning_rate = 0.0009
    #l1 = torch.prod(torch.tensor(observation_space.shape))
    #l2 = 150
    #l3 = action_space.n
    #return nn.Sequential(
    #    nn.Flatten(start_dim=-2, end_dim=-1),
    #    nn.Linear(l1, l2),
    #    nn.LeakyReLU(),
    #    nn.Linear(l2, l3))


def train(env_type_or_name='CartPole-v0', hidden_sizes=[32], lr=0.0009, 
          epochs=50, batch_size=5000, render=False):

    if issubclass(env_type_or_name, gym.Env):
        env = env_type_or_name()
    else:
        # make environment, check spaces, get obs / act dims
        env = gym.make(env_type_or_name)
        if env_type_or_name == 'CartPole-v0':
            env._max_episode_steps = 100_000

    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    save_file = 'trench_policy_network.pt'
    log_file = 'trench_policy_network.log'

    if os.path.exists(save_file):
        print(f'loading existing network from file: {save_file}')
        logits_net = torch.load(save_file)
    else:
        print(f'creating new network and saving to file: {save_file}')
        logits_net = my_mlp(env.observation_space, env.action_space)

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        policy = get_policy(obs)
        sample = policy.sample()
        return sample.item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32, device=device))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # Render the final state before resetting
                if (not finished_rendering_this_epoch) and render:
                    env.render()

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32, device=device),
                                  act=torch.as_tensor(np.array(batch_acts), dtype=torch.int32, device=device),
                                  weights=torch.as_tensor(np.array(batch_weights), dtype=torch.float32, device=device)
                                  )
        batch_loss.backward()
        optimizer.step()
        torch.save(logits_net, save_file)
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        with open(log_file, 'a') as f:
            info = 'epoch: %3d loss: %.3f return: %.3f ep_len: %.3f' % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
            print(info)
            f.write(info + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    train(env_type_or_name=TrenchRunnerEnv, render=args.render, lr=args.lr, epochs=200_000)
