import argparse
import os
import random
import sys
import time
from argparse import ArgumentParser
import os.path as osp

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium.spaces import Discrete

from agents.hdqn import MetaController, Controller
from .trainer import BaseTrainer
from utils import notice


def get_hdqn_config():
    parser = ArgumentParser()

    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
                        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
                        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
                        help="the frequency of training")

    return parser


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class HDQNTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.observation_space = self.envs.observation_space[0]
        self.cent_observation_space = self.envs.cent_observation_space[0]
        self.goal_space = Discrete(3)
        self.action_space = self.envs.action_space[0]

        args = config['all_args']
        self.lr = args.learning_rate

        # Construct meta-controller and controller
        self.meta_controller = MetaController(self.cent_observation_space, self.goal_space).to(self.device)
        self.target_meta_controller = MetaController(self.cent_observation_space, self.goal_space).to(self.device)
        self.target_meta_controller.load_state_dict(self.meta_controller.state_dict())
        self.controller = Controller(self.observation_space, self.action_space).to(self.device)
        self.target_controller = Controller(self.observation_space, self.action_space).to(self.device)
        self.target_controller.load_state_dict(self.controller.state_dict())

        # Construct the replay memory for meta-controller and controller
        self.meta_rb = ReplayBuffer(
            args.buffer_size,
            self.cent_observation_space,
            self.goal_space,
            self.device,
            handle_timeout_termination=False,
        )
        self.ctrl_rb = ReplayBuffer(
            args.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            handle_timeout_termination=False,
        )

    def train(self):
        # Construct the optimizers for meta-controller and controller
        meta_optimizer = optim.Adam(self.meta_controller.parameters(), lr=self.lr)
        ctrl_optimizer = optim.Adam(self.controller.parameters(), lr=self.lr)

        envs = self.envs
        args = self.all_args

        episodes = 0
        obs, cent_obs, _ = envs.reset()
        print(obs.shape)
        print(cent_obs.shape)
        # meta_obs = obs
        antenna = np.array([[[self.goal_space.sample()] for _ in envs.action_space] for _ in range(envs.num_envs)])
        extrinsic_rewards = 0

        pbar = trange(args.num_env_steps // args.n_rollout_threads)
        for step in pbar:
            steps = (step + 1) * args.n_rollout_threads
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.num_env_steps, steps)

            if step == 0:
                if random.random() < epsilon:
                    antenna = np.array([[[self.goal_space.sample()] for _ in envs.action_space] for _ in range(envs.num_envs)])
                else:
                    antenna = self.select_goal(cent_obs)

            goal_obs = np.concatenate([obs, antenna], axis=-1)

            if random.random() < epsilon:
                action = np.array([[s.sample() for s in envs.action_space] for _ in range(envs.num_envs)])
            else:
                action = self.select_action(goal_obs)

            actions = np.concatenate([antenna, action], axis=-1)
            next_obs, next_cent_obs, rewards, done, infos, _ = envs.step(actions)
            extrinsic_rewards += rewards

            # save data to replay buffer; handle `final_observation`
            next_goal_obs = np.concatenate([next_obs, antenna], axis=-1)
            self.ctrl_push(goal_obs, next_goal_obs, action, rewards, done)

            if (step + 1) % 10 == 0:
                self.meta_push(cent_obs, next_cent_obs, antenna, extrinsic_rewards, done)
                extrinsic_rewards = 0

                if random.random() < epsilon:
                    antenna = np.array([[[self.goal_space.sample()] for _ in envs.action_space] for _ in range(envs.num_envs)])
                else:
                    antenna = self.select_goal(next_cent_obs)
                # meta_obs = next_obs
                cent_obs = next_cent_obs

                if steps > args.learning_starts:
                    if (step + 1) % args.train_frequency == 0:
                        data = self.meta_rb.sample(args.batch_size)
                        with torch.no_grad():
                            target_meta_max, _ = self.target_meta_controller.net(data.next_observations).max(dim=1)
                            meta_td_target = data.rewards.flatten() + args.gamma * target_meta_max * (1 - data.dones.flatten())
                        # TODO: debug the following line
                        meta_old_val = self.meta_controller.gather(data.observations, data.actions).squeeze(dim=1)
                        meta_loss = F.mse_loss(meta_td_target, meta_old_val)

                        # if step % 100 == 0:
                        #     writer.add_scalar("losses/meta_td_loss", meta_loss, step)
                        #     writer.add_scalar("losses/meta_q_values", meta_old_val.mean().item(), step)
                        #     writer.add_scalar("charts/meta_SPS", int(step / (time.time() - start_time)), step)
                        #     pbar.set_postfix(SPS=int(step / (time.time() - start_time)), td_loss=meta_loss.item())

                        # optimize the model
                        meta_optimizer.zero_grad()
                        meta_loss.backward()
                        meta_optimizer.step()

            obs = next_obs

            # ALGO LOGIC: training.
            if steps > args.learning_starts:
                if (step + 1) % args.train_frequency == 0:
                    data = self.ctrl_rb.sample(args.batch_size)
                    with torch.no_grad():
                        target_max, _ = self.target_controller.net(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    # TODO: debug the following line
                    old_val = self.controller.gather(data.observations, data.actions)
                    loss = F.mse_loss(td_target, old_val)

                    # if step % 100 == 0:
                    #     writer.add_scalar("losses/td_loss", loss, step)
                    #     writer.add_scalar("losses/q_values", old_val.mean().item(), step)
                    #     writer.add_scalar("charts/SPS", int(step / (time.time() - start_time)), step)
                    #     pbar.set_postfix(SPS=int(step / (time.time() - start_time)), td_loss=loss.item())

                    # optimize the model
                    ctrl_optimizer.zero_grad()
                    loss.backward()
                    ctrl_optimizer.step()

                # update target network
                if steps % args.target_network_frequency == 0:
                    for targ_net_param, net_param in zip(self.target_controller.parameters(), self.controller.parameters()):
                        targ_net_param.data.copy_(
                            args.tau * net_param.data + (1.0 - args.tau) * targ_net_param.data
                        )
                    for targ_net_param, net_param in zip(self.target_meta_controller.parameters(), self.meta_controller.parameters()):
                        targ_net_param.data.copy_(
                            args.tau * net_param.data + (1.0 - args.tau) * targ_net_param.data
                        )

            if done.any():
                episodes += 1
                self.save()
                if episodes % self.log_interval == 0:
                    rew_df = pd.concat([pd.DataFrame(d['step_rewards']) for d in infos])
                    rew_info = rew_df.describe().loc[['mean', 'std', 'min', 'max']].unstack()
                    rew_info.index = ['_'.join(idx) for idx in rew_info.index]
                    self.log_train(rew_info, steps)


    def select_goal(self, cent_obs):
        # cent_obs = np.tile(cent_obs, (1, 7, 1))
        return np.array([self.meta_controller(torch.Tensor(ob).to(self.device))
                        .unsqueeze(-1).cpu().numpy() for ob in cent_obs])

    def select_action(self, obs):
        return np.array([self.controller(torch.Tensor(ob).to(self.device))
                        .cpu().numpy() for ob in obs])

    def meta_push(self, obs, next_obs, actions, rewards, done):
        obs = obs.reshape(-1, obs.shape[-1])
        obs1 = next_obs.reshape(-1, next_obs.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        rewards = np.repeat(rewards.reshape(-1, 1), self.num_agents, axis=1).reshape(-1)
        dones = np.repeat(done.reshape(-1, 1), self.num_agents, axis=1).reshape(-1)

        assert len(obs) == len(obs1) == len(actions) == len(rewards) == len(dones) \
               == self.envs.num_envs * self.num_agents

        for ob, ob1, a, r, d in zip(obs, obs1, actions, rewards, dones):
            self.meta_rb.add(ob, ob1, a, r, d, {})

    def ctrl_push(self, obs, next_obs, actions, rewards, done):
        obs = obs.reshape(-1, obs.shape[-1])
        obs1 = next_obs.reshape(-1, next_obs.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        rewards = np.repeat(rewards.reshape(-1, 1), self.num_agents, axis=1).reshape(-1)
        dones = np.repeat(done.reshape(-1, 1), self.num_agents, axis=1).reshape(-1)

        assert len(obs) == len(obs1) == len(actions) == len(rewards) == len(dones) \
               == self.envs.num_envs * self.num_agents

        for ob, ob1, a, r, d in zip(obs, obs1, actions, rewards, dones):
            self.ctrl_rb.add(ob, ob1, a, r, d, {})

    def take_actions(self, obs):
        return np.array([self.meta_controller(torch.Tensor(ob).to(self.device))
                         .cpu().numpy() for ob in obs])

    def save(self, version=''):
        notice("Saving models to {}".format(self.save_dir))
        torch.save(self.meta_controller.state_dict(), osp.join(self.save_dir, "meta%s.pt" % version))
        torch.save(self.controller.state_dict(), osp.join(self.save_dir, "sub%s.pt" % version))

    def load(self, version=''):
        path = os.path.join(self.model_dir, f"dqn{version}.pt")
        notice(f"Loading model from {path}")
        self.meta_controller.load_state_dict(torch.load(path))
        self.target_meta_controller.load_state_dict(self.meta_controller.state_dict())