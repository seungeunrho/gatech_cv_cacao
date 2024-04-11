import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
from gym.core import ObservationWrapper
from gymnasium.wrappers import GrayScaleObservation, FrameStack, AtariPreprocessing
import time
import collections
import cv2

class PPO(nn.Module):
    def __init__(self, config, device):
        super(PPO, self).__init__()
        self.data = []
        self.device = device
        self.n_update = 0
        self.config = config

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.fc_pi = nn.Sequential(
            nn.Linear(2560, 128),
            # nn.Linear(3200, 128),

            nn.ReLU(inplace=False),
            nn.Linear(128, 4)

        )

        self.fc_v = nn.Sequential(
            nn.Linear(2560, 128),
            # nn.Linear(3200, 128),

            nn.ReLU(inplace=False),
            nn.Linear(128, 1)
        )

        if self.config["add_aux_loss"]:
            self.fc_aux = nn.Sequential(
                nn.Linear(2560, 128),
                nn.ReLU(inplace=False),
                nn.Linear(128, 1)
            )

        self.optimizer = optim.Adam(self.parameters(), lr=self.config["learning_rate"])

    def _init_data(self):
        self.data = {
            "s":[],
            "a":[],
            "r":[],
            "s_prime":[],
            "done_mask":[],
            "prob_a":[]
        }

    def pi(self, x):
        x = self.cnn(x)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def v(self, x):
        x = self.cnn(x)
        v = self.fc_v(x)
        return v

    def reward_predict(self, x):
        x = self.cnn(x)
        r_pred = self.fc_aux(x)
        return r_pred

    def put_data(self, transition):
        self.data.append(transition)
        
    def choose_action(self, obs):
        with torch.no_grad():
            prob = self.pi(obs)
            m = Categorical(prob)
            a = m.sample()
        return a, prob

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, not_done_lst = [], [], [], [], [], []

        s, a, r, s_prime, prob_a, done = self.data[0]
        for i in range(len(s)):
            s_lst.append(torch.tensor(np.array(s[i]), dtype=torch.float, device=self.device))
            a_lst.append(torch.tensor(np.array(a[i]), device=self.device).unsqueeze(1))
            r_lst.append(torch.tensor(np.array(r[i]), device=self.device).unsqueeze(1))
            s_prime_lst.append(torch.tensor(s_prime[i], dtype=torch.float, device=self.device))
            prob_a_lst.append(torch.tensor(prob_a[i], device=self.device))
            done_mask = 1 - done
            not_done_lst.append(torch.tensor(done_mask[i], device=self.device).unsqueeze(1))

        self.data = []



        return s_lst, a_lst, r_lst, s_prime_lst, not_done_lst, prob_a_lst

    def train_net(self):
        s_lst, a_lst, r_lst, s_prime_lst, not_done_lst, prob_a_lst = self.make_batch()

        for i in range(self.config["K_epoch"]):
            for s, a, r, s_prime, done_mask, prob_a in zip(s_lst, a_lst, r_lst, s_prime_lst, not_done_lst, prob_a_lst):
                with torch.no_grad():
                    td_target = r + self.config["gamma"] * self.v(s_prime) * done_mask
                    delta = td_target - self.v(s)
                    delta = delta.to("cpu").detach().numpy()

                advantage_lst = []
                advantage = 0.0
                for delta_t in delta[::-1]:
                    advantage = self.config["gamma"] * self.config["lmbda"] * advantage + delta_t[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float, device=self.device, requires_grad=False)

                pi = self.pi(s)
                pi_a = pi.gather(1, a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.config["eps_clip"], 1 + self.config["eps_clip"]) * advantage
                critic_loss = F.smooth_l1_loss(self.v(s), td_target.detach()).mean()
                actor_loss = -torch.min(surr1, surr2).mean()
                loss = actor_loss + critic_loss

                aux_loss = 0.0
                if self.config["add_aux_loss"]:
                    r_pred = self.reward_predict(s)
                    aux_loss = F.smooth_l1_loss(r_pred, r.detach()).mean() * 0.1
                    loss += aux_loss

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                self.n_update += 1
        return actor_loss.item(), critic_loss.item(), aux_loss.item()