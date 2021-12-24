import gym
from bear.env_manager.manager import SubprocEnvManager
from bear.controller.base import Controller
from bear.data.experience import RolloutExperienceBuffer
from bear.policy.modelfree.pg import PGPolicy
from bear.policy.modelfree.ppo_clip import PPO2Policy
from bear.policy.modelfree.ppo_clip_a2c_refactor import PPO2A2C

from examples.common.env.atari_wrappers import wrap_deepmind
from examples.pong.adapter import PongAdapter
from examples.pong.exploration import PongExplorer
from examples.pong.model import ACModel

from torch.distributions import Categorical
import torch
import numpy as np
import time
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm


def run_evaluate_episodes(policy, adapter, explorer):
    avg_reward = []
    eval_episodes = 5
    test_env = wrap_deepmind(gym.make("PongNoFrameskip-v4"), dim=84, framestack=True, obs_format='NCHW')
    test_env.seed(int(time.time()))
    for _ in range(eval_episodes):
        obs = test_env.reset()
        done = False
        episode_reward = 0.
        while not done:
            obs = adapter.adapt_obs(obs)
            act_logits = policy.choose_action(np.asarray([obs]))

            action = explorer.reverse_action(act_logits)
            # print(action)
            next_obs, reward, done, info = test_env.step(action[0])
            test_env.render()
            episode_reward += reward
            obs = next_obs
        avg_reward.append(episode_reward)

    test_env.close()
    print("eval episode rewards:  ", avg_reward)
    return np.mean(avg_reward), np.min(avg_reward), np.max(avg_reward)


if __name__ == "__main__":
    logger = SummaryWriter('./logs/')
    training_num = 4
    seed = 100
    max_buffer_size = 160000
    feat_size = 4
    hidden_size = 128
    act_size = 6
    lr = 2e-4
    epochs = 1000000
    batch_size = 64
    best_avg_reward = -np.inf

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    adapter = PongAdapter()
    explorer = PongExplorer()
    buffer = RolloutExperienceBuffer(max_size=max_buffer_size)

    # model = ACModel(model_type="actor_only", hidden_size=hidden_size, feat_size=feat_size, act_size=act_size)
    # optim = torch.optim.Adam(model.actor.parameters(), lr=lr)
    # dist_fn = Categorical
    # #pg_policy = PGPolicy(model, optim, dist_fn=dist_fn, device=device, returns_norm=True)
    # pg_policy = PPO2Policy(model, optim, dist_fn, device=device,
    #                        returns_norm=True, eps_clip=0.2, dual_clip=1.5,
    #                        repeat_per_collect=10)

    model = ACModel(model_type="ac", feat_size=feat_size, act_size=act_size, hidden_size=hidden_size)
    #model.load('./checkpoints')
    model.set_device(device)

    optim = torch.optim.Adam(set(model.actor.parameters()).union(set(model.critic.parameters())),
                             lr=lr)
    dist_fn = Categorical

    # a2c_policy = A2CPolicy(model, optim, dist_fn,
    #                        value_loss_weight=0.5, entropy_loss_weight=0.001,
    #                        reward_norm=False, device=device)

    a2c_policy = PPO2A2C(model, optim, dist_fn, device=device,
                         returns_norm=True, eps_clip=0.2, gae_lambda=0.95,
                         repeat_per_collect=2, max_grad_norm=10)

    train_envs_manager = SubprocEnvManager([lambda: wrap_deepmind(gym.make("PongNoFrameskip-v4"), dim=84,
                                                                  framestack=True, obs_format='NCHW')
                                            for _ in range(training_num)])
    train_envs_manager.seed(seed)

    controller = Controller(a2c_policy, adapter, explorer, train_envs_manager, buffer)

    # epoch_start_time = time.time()
    for epoch in tqdm(range(epochs)):
        controller.collect(n_episode=10)
        loss = controller.learn(batch_size=batch_size)

        # epoch_end_time = time.time()
        # logger.add_scalar("epoch_cost_time", epoch_end_time - epoch_start_time, epoch)
        # print(epoch_end_time-epoch_start_time)
        # epoch_start_time = epoch_end_time

        if epoch % 20 == 0:
            print('---------------epoch: ', epoch, '--------------')
            avg_reward, min_reward, max_reward = run_evaluate_episodes(a2c_policy, adapter, explorer)
            print("avg_reward: ", avg_reward)
            print("loss: ", loss)
            logger.add_scalar("evaluate_avg5_reward", avg_reward, epoch)
            logger.add_scalar("evaluate_min5_reward", min_reward, epoch)
            logger.add_scalar("evaluate_max5_reward", max_reward, epoch)
            for k, v in loss.items():
                logger.add_scalar(k, v, epoch)
            if avg_reward >= best_avg_reward:
                best_avg_reward = avg_reward
                model.save('./checkpoints')
                print('best eval avg reward up to ', best_avg_reward)

    train_envs_manager.close()
