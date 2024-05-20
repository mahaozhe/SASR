"""
The Self-Adaptive Success Rate Shaping (SASR) algorithm, with the SAC as the backbone.

Main references:
- cleanrl docs: https://docs.cleanrl.dev/rl-algorithms/sac/
- cleanrl codes (sac continuous): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
"""

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.beta import Beta

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.buffers import ReplayBuffer

import os
import random
import datetime
import time
import math


class SASR:
    """
    The Self-Adaptive Success Rate Shaping (SASR) algorithm.
    """

    def __init__(self, env, actor_class, critic_class, exp_name="sasr", seed=1, cuda=0, gamma=0.99,
                 buffer_size=1000000, rb_optimize_memory=False, batch_size=256, policy_lr=3e-4, q_lr=1e-3, eps=1e-8,
                 alpha_lr=1e-4, target_network_frequency=1, tau=0.005, policy_frequency=2, alpha=0.2,
                 alpha_autotune=True, reward_weight=0.6, kde_bandwidth=0.2, kde_sample_burn_in=1000, rff_dim=1000,
                 retention_rate=0.1, write_frequency=100, save_folder="./sasr/"):
        """
        Initialize the SASR algorithm.
        :param env: the gymnasium-based environment
        :param actor_class: the actor class
        :param critic_class: the critic class
        :param exp_name: the name of the experiment
        :param seed: the random seed
        :param cuda: the cuda device
        :param gamma: the discount factor
        :param buffer_size: the size of the replay buffer
        :param rb_optimize_memory: whether to optimize the memory usage of the replay buffer
        :param batch_size: the batch size
        :param policy_lr: the learning rate of the policy network
        :param q_lr: the learning rate of the Q network
        :param eps: the epsilon for the Adam optimizer
        :param alpha_lr: the learning rate of the temperature parameter
        :param target_network_frequency: the target network update frequency
        :param tau: the soft update coefficient
        :param policy_frequency: the policy update frequency
        :param alpha: the temperature parameter
        :param alpha_autotune: whether to autotune the temperature parameter
        :param write_frequency: the write frequency
        :param save_folder: the folder to save the model
        :param reward_weight: the weight of the reward
        :param kde_bandwidth: the bandwidth of the KDE
        :param kde_sample_burn_in: the burn-in period of the KDE
        :param rff_dim: the dimension of the RFF
        :param retention_rate: the retention rate
        """

        self.exp_name = exp_name

        # set the random seeds
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        self.env = env

        # * for the SAC backbone
        # initialize the actor and critic networks
        self.actor = actor_class(self.env).to(self.device)
        self.qf_1 = critic_class(self.env).to(self.device)
        self.qf_2 = critic_class(self.env).to(self.device)
        self.qf_1_target = critic_class(self.env).to(self.device)
        self.qf_2_target = critic_class(self.env).to(self.device)

        # copy the parameters of the critic networks to the target networks
        self.qf_1_target.load_state_dict(self.qf_1.state_dict())
        self.qf_2_target.load_state_dict(self.qf_2.state_dict())

        # initialize the optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr, eps=eps)
        self.q_optimizer = optim.Adam(list(self.qf_1.parameters()) + list(self.qf_2.parameters()), lr=q_lr, eps=eps)

        # initialize the temperature parameter
        self.alpha_autotune = alpha_autotune
        if alpha_autotune:
            # set the target entropy as the negative of the action space dimension
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha

        # modify the observation space to be float32
        self.env.observation_space.dtype = np.float32
        # initialize the replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            optimize_memory_usage=rb_optimize_memory,
            handle_timeout_termination=False,
        )

        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        # * for the tensorboard writer
        run_name = "{}-{}-{}-{}".format(
            exp_name,
            env.unwrapped.spec.id,
            seed,
            datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S"),
        )
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))
        self.write_frequency = write_frequency

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

        # * for the SASR mechanism
        self.reward_weight = reward_weight / 2

        # * buffers to store the success and failure of states
        self.S_buffer = []
        self.S_buffer_tensor = torch.Tensor(self.S_buffer).to(self.device)
        self.F_buffer = []
        self.F_buffer_tensor = torch.Tensor(self.F_buffer).to(self.device)
        self.retention_rate = retention_rate

        self.kde_bandwidth = kde_bandwidth
        self.kde_sample_burn_in = kde_sample_burn_in
        self.obs_dim = env.observation_space.shape[0]

        # * sample the RFF mapping function
        self.rff_dim = rff_dim
        self.rff_W = torch.randn(rff_dim, self.obs_dim).to(self.device) / kde_bandwidth
        self.rff_b = torch.rand(rff_dim).to(self.device) * 2 * torch.pi

    def update_S(self, trajectory):
        retention_interval = int(1 / self.retention_rate) + 1
        if retention_interval >= len(trajectory):
            return
        trajectory = trajectory[::retention_interval]

        self.S_buffer += trajectory
        self.S_buffer_tensor = torch.Tensor(np.array(self.S_buffer)).to(self.device)

    def update_F(self, trajectory):
        retention_interval = int(1 / self.retention_rate) + 1
        if retention_interval >= len(trajectory):
            return
        trajectory = trajectory[::retention_interval]

        self.F_buffer += trajectory
        self.F_buffer_tensor = torch.Tensor(np.array(self.F_buffer)).to(self.device)

    def KDE_RFF_sample(self, buffer, batch):

        if buffer.shape[0] <= self.kde_sample_burn_in:
            return torch.zeros(batch.shape[0]).to(self.device)

        z_buffer = math.sqrt(2 / self.rff_dim) * torch.cos(torch.matmul(buffer, self.rff_W.T) + self.rff_b)
        z_batch = math.sqrt(2 / self.rff_dim) * torch.cos(torch.matmul(batch, self.rff_W.T) + self.rff_b)

        kde_estimates = torch.sum(torch.matmul(z_buffer, z_batch.T) ** 2, dim=0)

        return kde_estimates

    def learn(self, total_timesteps=1000000, learning_starts=5000):

        obs, _ = self.env.reset()

        # * to record the trajectory
        trajectory = []
        reward_positive = False

        for global_step in range(total_timesteps):
            if global_step < learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _, _ = self.actor.get_action(torch.Tensor(np.expand_dims(obs, axis=0)).to(self.device))
                action = action.detach().cpu().numpy()[0]

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if "episode" in info:
                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

            self.replay_buffer.add(obs, next_obs, action, reward, done, info)
            trajectory.append(obs)

            if reward > 0:
                reward_positive = True
            elif reward == 0 and reward_positive:
                # * use the trajectory to update the Success buffer
                self.update_S(trajectory)
                trajectory = []
                reward_positive = False

            if not done:
                obs = next_obs
            else:
                obs, _ = self.env.reset()

                # * record the trajectory
                if reward_positive:
                    self.update_S(trajectory)
                else:
                    self.update_F(trajectory)

                trajectory = []
                reward_positive = False

            if global_step > learning_starts:
                self.optimize(global_step)

        self.env.close()
        self.writer.close()

    def optimize(self, global_step):
        data = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
            qf_1_next_target = self.qf_1_target(data.next_observations, next_state_actions)
            qf_2_next_target = self.qf_2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf_1_next_target, qf_2_next_target) - self.alpha * next_state_log_pi

            # * to calculate the SASR reward
            density_values_S = self.KDE_RFF_sample(self.S_buffer_tensor, data.observations)
            density_values_F = self.KDE_RFF_sample(self.F_buffer_tensor, data.observations)
            shaped_rewards = Beta(density_values_S + 1, density_values_F + 1).sample()

            sasr_rewards = data.rewards.flatten() + self.reward_weight * shaped_rewards
            next_q_value = sasr_rewards + (1 - data.dones.flatten()) * self.gamma * min_qf_next_target.view(-1)

        qf_1_a_values = self.qf_1(data.observations, data.actions).view(-1)
        qf_2_a_values = self.qf_2(data.observations, data.actions).view(-1)
        qf_1_loss = F.mse_loss(qf_1_a_values, next_q_value)
        qf_2_loss = F.mse_loss(qf_2_a_values, next_q_value)
        qf_loss = qf_1_loss + qf_2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if global_step % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                pi, log_pi, _ = self.actor.get_action(data.observations)
                qf_1_pi = self.qf_1(data.observations, pi)
                qf_2_pi = self.qf_2(data.observations, pi)
                min_qf_pi = torch.min(qf_1_pi, qf_2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.alpha_autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(data.observations)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # update the target networks
        if global_step % self.target_network_frequency == 0:
            for param, target_param in zip(self.qf_1.parameters(), self.qf_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf_2.parameters(), self.qf_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if global_step % self.write_frequency == 0:
            self.writer.add_scalar("losses/qf_1_values", qf_1_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/qf_2_values", qf_2_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/qf_1_loss", qf_1_loss.item(), global_step)
            self.writer.add_scalar("losses/qf_2_loss", qf_2_loss.item(), global_step)
            self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            self.writer.add_scalar("losses/alpha", self.alpha, global_step)
            if self.alpha_autotune:
                self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    def save(self, indicator="best"):
        torch.save(self.actor.state_dict(),
                   os.path.join(self.save_folder, "actor-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
        torch.save(self.qf_1.state_dict(),
                   os.path.join(self.save_folder, "qf_1-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
        torch.save(self.qf_2.state_dict(),
                   os.path.join(self.save_folder, "qf_2-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
