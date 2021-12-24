import gym
from bear.env_manager.manager import SubprocEnvManager
from bear.controller.base import Controller
from bear.data.experience import RolloutExperienceBuffer
from examples.cartpole.adapter import CartPoleAdapter
from examples.cartpole.exploration import CartPoleExplorer
# from examples.cartpole.model import MLPActor, MLPExtractor
from bear.policy.modelfree.pg import PGPolicy
from bear.policy.modelfree.ppo_clip import PPO2Policy

from examples.cartpole.model import ACModel
from torch.distributions import Categorical
import torch
import numpy as np
import time
import os
from tensorboardX import SummaryWriter


def run_evaluate_episodes(policy, adapter, explorer):
    avg_reward = []
    eval_episodes = 5
    test_env = gym.make('CartPole-v0')
    test_env.seed(int(time.time()))
    for _ in range(eval_episodes):
        obs = test_env.reset()
        done = False
        episode_reward = 0.
        while not done:
            obs = adapter.adapt_obs(obs)
            act_logits = policy.choose_action(np.asarray([obs]))

            action = explorer.reverse_action(act_logits)
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
    training_num = 6
    seed = 10
    max_buffer_size = 1000000
    feat_size = 4
    hidden_size = 128
    act_size = 2
    lr = 1e-2
    epochs = 1000000
    best_avg_reward = 0.

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    adapter = CartPoleAdapter()
    explorer = CartPoleExplorer()
    buffer = RolloutExperienceBuffer(max_size=max_buffer_size)

    # extractor = MLPExtractor(feat_size=feat_size, hidden_size=hidden_size)
    # actor_model = MLPActor(extractor,
    #                        hidden_size=hidden_size,
    #                        act_size=act_size)
    # actor_model.load_state_dict(torch.load('./checkpoints/actor1.pt', map_location=torch.device('cpu')))
    model = ACModel(model_type="actor_only", hidden_size=hidden_size, feat_size=feat_size, act_size=act_size)
    optim = torch.optim.Adam(model.actor.parameters(), lr=lr)
    dist_fn = Categorical
    # pg_policy = PGPolicy(model, optim, dist_fn=dist_fn, device=device, returns_norm=True)
    pg_policy = PPO2Policy(model, optim, dist_fn, device=device,
                           returns_norm=True, eps_clip=0.2, dual_clip=1.5, value_clip=10,
                           repeat_per_collect=10)

    train_envs_manager = SubprocEnvManager([lambda: gym.make("CartPole-v0") for _ in range(training_num)])
    train_envs_manager.seed(seed)

    controller = Controller(pg_policy, adapter, explorer, train_envs_manager, buffer)

    for epoch in range(epochs):
        controller.collect(n_episode=1)
        loss = controller.learn()

        if epoch % 100 == 0:
            print('---------------epoch: ', epoch, '--------------')
            avg_reward, min_reward, max_reward = run_evaluate_episodes(pg_policy, adapter, explorer)
            print("avg_reward: ", avg_reward)
            print("loss: ", loss)
            logger.add_scalar("evaluate_avg5_reward", avg_reward, epoch)
            logger.add_scalar("evaluate_min5_reward", min_reward, epoch)
            logger.add_scalar("evaluate_max5_reward", max_reward, epoch)
            for k, v in loss.items():
                logger.add_scalar(k, v, epoch)
            if avg_reward >= best_avg_reward:
                best_avg_reward = avg_reward
                # torch.save(model.actor.state_dict(), os.path.join('./checkpoints', 'actor1.pt'))
                print('best eval avg reward up to ', best_avg_reward)

    train_envs_manager.close()
