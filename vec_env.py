import gymnasium as gym

import numpy as np
from gym.core import ObservationWrapper
from gymnasium.wrappers import GrayScaleObservation, FrameStack, AtariPreprocessing
import time
import collections
import cv2 
from skimage.metrics import mean_squared_error,peak_signal_noise_ratio,structural_similarity
import matplotlib.pyplot as plt

class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (84, 84)

    def observation(self, img):
        """what happens to each observation"""

        img = img.astype('float32') / 255.
        img = img[14:80, :]
        # img = img.reshape(1, 84, 84)

        return img

###### Edge Detection https://www.analyticsvidhya.com/blog/2022/08/comprehensive-guide-to-edge-detection-algorithms/ #######
class EdgeDetection(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self, env)

    def observation(self, img):
        """what happens to each observation"""

        img1 = (img * 255).astype(np.uint8)
        
        # Enhance contrast using histogram equalization on the V channel of the HSV image
        hsv_img = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        hsv_img[:, :, 2] = cv2.equalizeHist(hsv_img[:, :, 2])
        img_enhanced = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        
        # Convert enhanced image to grayscale
        gray_img = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detector with adjusted thresholds
        edges = cv2.Canny(gray_img, 5, 100)  # Example: lower threshold is reduced
        
        # # Example: replacing one channel with edges, adjust as needed
        # img1[:, :, 2] = edges
        # gray_img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        # fig, ax = plt.subplots(1, 2, figsize=(18, 18))
        # ax[0].imshow(img, cmap='gray')
        # ax[1].imshow(gray_img, cmap='gray')
        # for a in ax:
        #     a.axis('off')
        # # Save the figure to a file
        # fig.savefig('image/image.png')
        # plt.close(fig)  # Close the figure to free memory
        return img1
###########

class VecEnv:
    def __init__(self, n_env=10, step_limit=1000):
        self.env_lst = []
        self.score_lst = []
        self.step_lst = []
        self.step_limit = step_limit
        self.score_buffer = collections.deque(maxlen=n_env)
        self.n_epi = 0
        for i in range(n_env):
            env = gym.make("Breakout-v4", obs_type="rgb", frameskip=1)
            env = EdgeDetection(env)
            env = AtariPreprocessing(env)
            env = PreprocessAtari(env)
            env = FrameStack(env, num_stack=4)

            self.env_lst.append(env)
            self.score_lst.append(0.)
            self.step_lst.append(0)

    def step(self, action):
        s_lst, r_lst, done_lst = [], [], []
        for i in range(len(self.env_lst)):
            # random_action = self.env_lst[i].action_space.sample()
            # s_prime, r, done, truncated, info = self.env_lst[i].step(random_action)

            s_prime, r, done, truncated, info = self.env_lst[i].step(action[i])
            done_lst.append(done)
            self.score_lst[i] += r
            self.step_lst[i] += 1


            if done or self.step_lst[i] > self.step_limit:
                self.n_epi += 1
                s_prime, _ = self.env_lst[i].reset()
                self.step_lst[i] = 0
                self.score_buffer.append(self.score_lst[i])
                self.score_lst[i] = 0.

            s_lst.append(s_prime)
            r_lst.append(r)

        return np.array(s_lst), np.array(r_lst), np.array(done_lst)

    def reset_all(self):
        s_lst = []
        for i in range(len(self.env_lst)):
            s_prime, _ = self.env_lst[i].reset()
            s_lst.append(s_prime)

        return np.array(s_lst)