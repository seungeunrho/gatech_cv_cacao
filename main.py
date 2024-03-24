import torch
from torch.distributions import Categorical

import numpy as np
from vec_env import VecEnv
import time
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO
import os
from datetime import datetime




def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    now = datetime.now()
    dt_string = now.strftime("%m%d-%H_%M_%S")
    save_dir = os.path.join(config["log_dir"], "log_"+dt_string)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(log_dir=save_dir, flush_secs=10)


    env = VecEnv(n_env=config["n_env"], step_limit=config["step_limit"])
    model = PPO(config, device).to(device)
    score = 0.0
    best_score = -5000.0

    t1 = time.time()
    s = env.reset_all()
    for iter_num in range(config["train_iter"]):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for t in range(config["T_horizon"]):
            with torch.no_grad():
                tensor_s = torch.from_numpy(s).float().to(device)
                a, prob = model.choose_action(tensor_s)
                a_numpy = a.cpu().numpy()

            s_prime, r, done = env.step(a_numpy)
            prob_a = torch.gather(prob, 1, a.unsqueeze(1)).cpu().numpy()

            s_lst.append(s)
            a_lst.append(a_numpy)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_lst.append(done)

            s = s_prime


        model.put_data((np.stack(s_lst).transpose(1, 0, 2, 3, 4),
                        np.stack(a_lst).transpose(1, 0),
                        np.stack(r_lst).transpose(1, 0),
                        np.stack(s_prime_lst).transpose(1, 0, 2, 3, 4),
                        np.stack(prob_a_lst).transpose(1, 0, 2),
                        np.stack(done_lst).transpose(1, 0)))

        model.train_net()


        if iter_num % config["print_interval"] == 0:
            score = np.mean(env.score_buffer)

            cur_time = (time.time() - t1) / 60.0
            writer.add_scalar('n_epi', env.n_epi, iter_num)
            writer.add_scalar('n_update', model.n_update, iter_num)
            writer.add_scalar('score', score, iter_num)
            print("Iter:{}, n_epi:{}, n_update:{}, score:{:.1f}, mean_step:{:.1f}, time: {:.1f}mins".format(
                iter_num, env.n_epi, model.n_update, score, np.mean(env.step_lst), cur_time))

            if score > best_score:
                model_path = os.path.join(save_dir, f"model_iter{iter_num}_score{score}.pt")
                torch.save(model, model_path)
                best_score = score
                print(f"new model saved. current best score: {score} ")


if __name__ == '__main__':
    config = {
        "learning_rate": 0.0001,
        "log_dir" : "logs",
        "gamma": 0.98,
        "lmbda": 0.95,
        "eps_clip": 0.1,
        "K_epoch": 3,
        "T_horizon": 20,
        "n_env": 24,
        "step_limit": 500,
        "print_interval": 10,
        "train_iter": 100000,
    }
    main(config)

