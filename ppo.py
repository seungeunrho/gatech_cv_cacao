import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from vit_pytorch import ViT
import numpy as np
from gym.core import ObservationWrapper
from gymnasium.wrappers import GrayScaleObservation, FrameStack, AtariPreprocessing
import time
import collections
import cv2
from torch.optim import Adam
class PPO(nn.Module):
    def __init__(self, config, device):
        super(PPO, self).__init__()
        self.data = []
        self.device = device
        self.n_update = 0
        self.config = config
        self.vit = ViT(
                image_size=(66, 84),  # Adjusted image size
                patch_size=6,   # Adjusted patch size
                num_classes = 1,
                dim = 128,#256,
                depth = 2,
                heads = 2,
                mlp_dim = 256,
                channels= 4,
                dropout = 0.1,
                emb_dropout = 0.1
            ).to(device)
        self.vit_pi = ViT(
                image_size=(66, 84),  # Adjusted image size
                patch_size=6,   # Adjusted patch size
                num_classes = 4,
                dim = 128,#256,
                depth = 2,
                heads = 2,
                mlp_dim = 256,
                channels= 4,
                dropout = 0.1,
                emb_dropout = 0.1
            ).to(device)        
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
    def ViT_model(self,x):
        return self.vit(x)
    def ViT_pi(self,x):
        return self.vit_pi(x)
    def pi(self, x):
        x = self.cnn(x)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def v(self, x):
        x = self.cnn(x)
        v = self.fc_v(x)
        return v
    def pi_vit(self,x):
        return self.vit(x)
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
                    if self.config["image_encoder"] == "cnn":
                        delta = td_target - self.v(s)
                    elif self.config["image_encoder"] == "vit":
                        delta = td_target - self.ViT_model(s)
                    # delta = td_target - self.v(s)
                    delta = delta.to("cpu").detach().numpy()
                # print(prob_a.shape)
                advantage_lst = []
                advantage = 0.0
                for delta_t in delta[::-1]:
                    advantage = self.config["gamma"] * self.config["lmbda"] * advantage + delta_t[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float, device=self.device, requires_grad=False)
                if self.config["image_encoder"] == "cnn":
                    pi = self.pi(s)
                elif self.config["image_encoder"] == "vit":
                    pi = self.vit_pi(s)
                # pi = self.pi(s)
                pi_a = pi.gather(1, a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.config["eps_clip"], 1 + self.config["eps_clip"]) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                self.n_update += 1

    def train_net_vit(self):
        s_lst, a_lst, r_lst, s_prime_lst, not_done_lst, prob_a_lst = self.make_batch()

        for i in range(self.config["K_epoch"]):
            for s, a, r, s_prime, done_mask, prob_a in zip(s_lst, a_lst, r_lst, s_prime_lst, not_done_lst, prob_a_lst):
                with torch.no_grad():
                    td_target = r + self.config["gamma"] * self.v(s_prime) * done_mask
                    # print(s.shape)
                    # print(td_target.shape)                 
                       # batch_size, channels, height, width = s.shape
                    # patch_size = 6
                    # input_data = input_data = s.view(batch_size, channels * (height // patch_size) * (width // patch_size), patch_size ** 2)
                    delta = td_target - self.ViT_model(s)
                    delta = delta.to("cpu").detach().numpy()
                # print(prob_a.shape)
                advantage_lst = []
                advantage = 0.0
                for delta_t in delta[::-1]:
                    advantage = self.config["gamma"] * self.config["lmbda"] * advantage + delta_t[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float, device=self.device, requires_grad=False)

                # pi = self.pi(s)
                pi = self.vit_pi(s)
                # print(pi.shape)
                pi_a = pi.gather(1, a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.config["eps_clip"], 1 + self.config["eps_clip"]) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                self.n_update += 1