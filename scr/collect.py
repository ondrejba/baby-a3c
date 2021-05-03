import argparse
import gym
import numpy as np
import torch
import torch.nn.functional as F
from baby_a3c import NNPolicy, prepro


def main(args):

    hidden = 256
    max_episodes = args.max_episodes
    min_burnin = args.min_burnin
    max_burnin = args.max_burnin
    env_name = args.env_id
    save_dir = "./{:s}/".format(env_name.lower())

    torch.manual_seed(0)

    env = gym.make(env_name)
    env.seed(0)
    num_actions = env.action_space.n  # get the action space of this game

    model = NNPolicy(channels=1, memsize=hidden, num_actions=num_actions)
    step = model.try_load(save_dir)
    assert step != 0

    ep_steps = []
    episode_length, epr, eploss, done = 0, 0, 0, True
    state = torch.tensor(prepro(env.reset()))
    hx = torch.zeros(1, 256)

    with torch.no_grad():

        while True:

            episode_length += 1
            value, logit, hx = model((state.view(1, 1, 80, 80), hx))
            logp = F.log_softmax(logit, dim=-1)
            action = torch.exp(logp).multinomial(num_samples=1).data[0]

            state, reward, done, _ = env.step(action.numpy()[0])

            state = torch.tensor(prepro(state))
            epr += reward
            done = done or episode_length >= 1e4

            if done:
                hx = torch.zeros(1, 256)
                print("episode done; steps: {:d}, return {:.2f}".format(episode_length, epr))
                ep_steps.append(episode_length)
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(prepro(env.reset()))

                if len(ep_steps) >= max_episodes:
                    break

    print("episode steps, min: {:.1f}, max: {:.1f}, mean: {:.1f}".format(
        np.min(ep_steps), np.max(ep_steps), np.mean(ep_steps)
    ))


parser = argparse.ArgumentParser()
parser.add_argument("env_id")
parser.add_argument("min_burnin", type=int)
parser.add_argument("max_burnin", type=int)
parser.add_argument("max_episodes", type=int)
parsed = parser.parse_args()
main(parsed)
